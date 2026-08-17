package expo.modules.location.taskConsumers

import android.app.PendingIntent
import android.app.job.JobParameters
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.location.Location
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.Looper
import android.os.PersistableBundle
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationAvailability
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import expo.modules.core.MapHelper
import expo.modules.core.arguments.MapArguments
import expo.modules.core.arguments.ReadableArguments
import expo.modules.core.interfaces.Arguments
import expo.modules.core.interfaces.LifecycleEventListener
import expo.modules.interfaces.taskManager.TaskConsumer
import expo.modules.interfaces.taskManager.TaskConsumerInterface
import expo.modules.interfaces.taskManager.TaskExecutionCallback
import expo.modules.interfaces.taskManager.TaskInterface
import expo.modules.interfaces.taskManager.TaskManagerUtilsInterface
import expo.modules.location.AppForegroundedSingleton
import expo.modules.location.LocationHelpers
import expo.modules.location.records.LocationOptions
import expo.modules.location.records.LocationResponse
import expo.modules.location.services.LocationTaskService
import expo.modules.location.services.LocationTaskService.ServiceBinder
import kotlin.math.abs

/**
 * Patch ATMR : FGS via LocationCallback (Android 14+/16) + instrumentation RCA P1–P8.
 * Tags logcat : ATMR_LTC_P1 … ATMR_LTC_P8 (filtre adb: `ATMR_LTC_P`).
 */
class LocationTaskConsumer(context: Context, taskManagerUtils: TaskManagerUtilsInterface?) : TaskConsumer(context, taskManagerUtils), TaskConsumerInterface, LifecycleEventListener {
  private var mTask: TaskInterface? = null
  private var mPendingIntent: PendingIntent? = null
  private var mService: LocationTaskService? = null
  private var mLocationRequest: LocationRequest? = null
  private var mLocationCallback: LocationCallback? = null
  private var mLastReportedLocation: Location? = null
  private var mDeferredDistance = 0.0
  private val mDeferredLocations: MutableList<Location> = ArrayList()
  private var mIsHostPaused = true
  private val mLocationClient: FusedLocationProviderClient by lazy {
    LocationServices.getFusedLocationProviderClient(context)
  }

  //region TaskConsumerInterface
  override fun taskType(): String {
    return "location"
  }

  override fun didRegister(task: TaskInterface) {
    mTask = task
    if (shouldUseForegroundService(task.options)) {
      maybeStartForegroundService()
    } else {
      startLocationUpdates()
    }
  }

  override fun didUnregister() {
    stopLocationUpdates()
    stopForegroundService()
    mTask = null
    mPendingIntent = null
    mLocationRequest = null
  }

  override fun setOptions(options: Map<String, Any>) {
    super.setOptions(options)

    stopLocationUpdates()
    if (shouldUseForegroundService(options as Map<String?, Any?>)) {
      maybeStartForegroundService()
    } else {
      startLocationUpdates()
    }
  }

  override fun didReceiveBroadcast(intent: Intent) {
    // P1 — PendingIntent path (non-FGS / legacy)
    Log.i(DIAG, "P1 didReceiveBroadcast received=true action=${intent.action}")
    mTask ?: run {
      Log.w(DIAG, "P1 abort task=null")
      return
    }
    val result = LocationResult.extractResult(intent)
    if (result != null) {
      val locations = result.locations
      val first = locations.firstOrNull()
      // P2
      Log.i(
        DIAG,
        "P2 LocationResult null=false size=${locations.size} " +
          "time=${first?.time} ertNanos=${first?.elapsedRealtimeNanos} " +
          "lat=${first?.latitude} lon=${first?.longitude}"
      )
      deferLocations(locations)
      maybeReportDeferredLocations()
    } else {
      Log.w(DIAG, "P2 LocationResult null=true → fallback lastLocation")
      try {
        mLocationClient.lastLocation.addOnCompleteListener { task ->
          val loc = task.result
          val ageMs = if (loc != null) {
            SystemClock.elapsedRealtime() - (loc.elapsedRealtimeNanos / 1_000_000L)
          } else {
            -1L
          }
          // P3
          Log.i(
            DIAG,
            "P3 fallback lastLocation null=${loc == null} ageMs=$ageMs " +
              "time=${loc?.time} lat=${loc?.latitude} lon=${loc?.longitude}"
          )
          loc?.let {
            deferLocations(listOf(it))
            maybeReportDeferredLocations()
          }
        }
      } catch (e: SecurityException) {
        Log.e(TAG, "Cannot get last location: " + e.message)
        Log.e(DIAG, "P3 SecurityException ${e.message}")
      }
    }
  }

  override fun didExecuteJob(jobService: JobService, params: JobParameters): Boolean {
    val data = taskManagerUtils.extractDataFromJobParams(params)
    val locationBundles = ArrayList<Bundle>()
    for (persistableLocationBundle in data) {
      val locationBundle = Bundle()
      val coordsBundle = Bundle()
      if (persistableLocationBundle != null) {
        coordsBundle.putAll(persistableLocationBundle.getPersistableBundle("coords"))
        locationBundle.putAll(persistableLocationBundle)
        locationBundle.putBundle("coords", coordsBundle)
        locationBundles.add(locationBundle)
      }
    }
    // P7
    Log.i(DIAG, "P7 didExecuteJob bundles.size=${locationBundles.size} rawData.size=${data.size}")
    executeTaskWithLocationBundles(locationBundles) { jobService.jobFinished(params, false) }

    // Returning `true` indicates that the job is still running, but in async mode.
    // In that case we're obligated to call `jobService.jobFinished` as soon as the async block finishes.
    return true
  }

  //region private
  private fun startLocationUpdates() {
    val context = context ?: run {
      Log.w(TAG, "The context has been abandoned")
      return
    }

    if (!LocationHelpers.isAnyProviderAvailable(context)) {
      Log.w(TAG, "There is no location provider available")
      return
    }
    val task = mTask ?: run {
      Log.w(TAG, "Could not find a location task for the location update")
      return
    }
    mLocationRequest = LocationHelpers.prepareLocationRequest(LocationOptions(task.options))
    val locationRequest = mLocationRequest ?: run {
      Log.w(TAG, "Could not find a location request for the location update")
      return
    }

    if (shouldUseForegroundService(task.options)) {
      startLocationUpdatesWithCallback(locationRequest)
      return
    }

    mPendingIntent = preparePendingIntent()
    val intent = mPendingIntent ?: run {
      Log.w(TAG, "Could not find intent for the location update")
      return
    }

    try {
      mLocationClient.requestLocationUpdates(locationRequest, intent)
      Log.i(DIAG, "P0 requestLocationUpdates PendingIntent path")
    } catch (e: SecurityException) {
      Log.w(TAG, "Location request has been rejected.", e)
    }
  }

  private fun startLocationUpdatesWithCallback(locationRequest: LocationRequest) {
    stopLocationCallbackUpdates()
    val callback = object : LocationCallback() {
      override fun onLocationResult(locationResult: LocationResult) {
        val locations = locationResult.locations
        val first = locations.firstOrNull()
        // P2b — chemin FGS LocationCallback (ATMR), équivalent extractResult
        Log.i(
          DIAG,
          "P2b onLocationResult size=${locations.size} " +
            "time=${first?.time} ertNanos=${first?.elapsedRealtimeNanos} " +
            "lat=${first?.latitude} lon=${first?.longitude} " +
            "sLastTimestamp=$sLastTimestamp"
        )
        if (locations.isNotEmpty()) {
          deferLocations(locations)
          maybeReportDeferredLocations()
        } else {
          Log.w(DIAG, "P2b empty locations — no defer")
        }
      }

      override fun onLocationAvailability(locationAvailability: LocationAvailability) {
        val available = locationAvailability.isLocationAvailable
        // P2c — origine exacte du log historique « Location unavailable… »
        Log.i(
          DIAG,
          "P2c onLocationAvailability isLocationAvailable=$available " +
            "deferred.size=${mDeferredLocations.size} hostPaused=$mIsHostPaused"
        )
        if (!available) {
          Log.w(TAG, "Location unavailable for foreground-service task delivery")
        }
      }
    }
    mLocationCallback = callback
    try {
      mLocationClient.requestLocationUpdates(locationRequest, callback, Looper.getMainLooper())
      Log.i(TAG, "Started location updates via LocationCallback (FGS path)")
      Log.i(DIAG, "P0 requestLocationUpdates LocationCallback FGS path")
    } catch (e: SecurityException) {
      Log.w(TAG, "Location callback request has been rejected.", e)
      Log.e(DIAG, "P0 SecurityException ${e.message}")
    }
  }

  private fun stopLocationCallbackUpdates() {
    mLocationCallback?.let {
      try {
        mLocationClient.removeLocationUpdates(it)
      } catch (e: SecurityException) {
        Log.w(TAG, "Failed to remove location callback updates", e)
      }
    }
    mLocationCallback = null
  }

  private fun stopLocationUpdates() {
    stopLocationCallbackUpdates()
    mPendingIntent?.let {
      mLocationClient.removeLocationUpdates(it)
      it.cancel()
    }
    mPendingIntent = null
  }

  private fun maybeStartForegroundService() {
    // Foreground service is available as of Android Oreo.
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
      startLocationUpdates()
      return
    }
    if (!AppForegroundedSingleton.isForegrounded) {
      Log.w(TAG, "Foreground location task cannot be started while the app is in the background!")
      return
    }
    val task = mTask ?: run {
      Log.w(TAG, "Location task is null")
      return
    }
    val options: ReadableArguments = MapArguments(task.options)
    val useForegroundService = shouldUseForegroundService(task.options)

    // Service is already running, but the task has been registered again without `foregroundService` option.
    if (mService != null && !useForegroundService) {
      stopForegroundService()
      return
    }

    // Service is not running and the user don't want to start foreground service.
    if (!useForegroundService) {
      return
    }

    val serviceOptions = options.getArguments(FOREGROUND_SERVICE_KEY).toBundle()

    // Foreground service is requested but not running.
    if (mService == null) {
      val serviceIntent = Intent(context, LocationTaskService::class.java)
      val extras = Bundle()

      // extras param name is appId for legacy reasons
      extras.putString("appId", task.appScopeKey)
      extras.putString("taskName", task.name)
      extras.putBoolean("killService", serviceOptions.getBoolean("killServiceOnDestroy", false))
      serviceIntent.putExtras(extras)
      context.startForegroundService(serviceIntent)
      context.bindService(
        serviceIntent,
        object : ServiceConnection {
          override fun onServiceConnected(name: ComponentName, service: IBinder) {
            mService = (service as? ServiceBinder)?.service
            mService?.let {
              it.setParentContext(context)
              it.startForeground(serviceOptions)
              // patch(atmr): Android 14+/16 — requête OS après FGS actif, via LocationCallback.
              startLocationUpdates()
            }
          }

          override fun onServiceDisconnected(name: ComponentName) {
            mService?.stop()
            mService = null
          }
        },
        Context.BIND_AUTO_CREATE
      )
    } else {
      // Restart the service with new service options.
      mService?.startForeground(serviceOptions)
      startLocationUpdates()
    }
  }

  private fun stopForegroundService() {
    mService?.stop()
  }

  private fun deferLocations(locations: List<Location>) {
    val size = mDeferredLocations.size
    var lastLocation = if (size > 0) mDeferredLocations[size - 1] else mLastReportedLocation
    for (location in locations) {
      if (lastLocation != null) {
        mDeferredDistance += abs(location.distanceTo(lastLocation)).toDouble()
      }
      lastLocation = location
    }
    mDeferredLocations.addAll(locations)
  }

  private fun maybeReportDeferredLocations() {
    val should = shouldReportDeferredLocations()
    // P4
    Log.i(
      DIAG,
      "P4 defer/report deferred.size=${mDeferredLocations.size} " +
        "hostPaused=$mIsHostPaused shouldReport=$should " +
        "deferredDistance=$mDeferredDistance sLastTimestamp=$sLastTimestamp"
    )
    if (!should) {
      return
    }
    val context = context.applicationContext
    val data: MutableList<PersistableBundle> = ArrayList()
    var accepted = 0
    var rejectedTs = 0
    for (location in mDeferredLocations) {
      val timestamp = location.time
      val ok = timestamp > sLastTimestamp
      // P5
      Log.i(
        DIAG,
        "P5 filter incoming.time=$timestamp sLastTimestamp=$sLastTimestamp accepted=$ok " +
          "lat=${location.latitude} lon=${location.longitude}"
      )
      if (ok) {
        val bundle = LocationResponse(location).toBundle(PersistableBundle::class.java)
        data.add(bundle)
        sLastTimestamp = timestamp
        accepted++
      } else {
        rejectedTs++
      }
    }
    Log.i(DIAG, "P5 summary accepted=$accepted rejectedTs=$rejectedTs bundles=${data.size}")
    if (data.size > 0) {
      // Save last reported location, reset the distance and clear a list of locations.
      mLastReportedLocation = mDeferredLocations[mDeferredLocations.size - 1]
      mDeferredDistance = 0.0
      mDeferredLocations.clear()

      if (mService != null && mLocationCallback != null) {
        val locationBundles = ArrayList<Bundle>()
        for (persistableLocationBundle in data) {
          val locationBundle = Bundle()
          val coordsBundle = Bundle()
          coordsBundle.putAll(persistableLocationBundle.getPersistableBundle("coords"))
          locationBundle.putAll(persistableLocationBundle)
          locationBundle.putBundle("coords", coordsBundle)
          locationBundles.add(locationBundle)
        }
        // P6 — chemin direct FGS (pas JobScheduler)
        Log.i(DIAG, "P6 directFGS execute bundles.size=${locationBundles.size}")
        executeTaskWithLocationBundles(locationBundles) { /* direct FGS path */ }
        return
      }

      // P6 — JobScheduler
      Log.i(DIAG, "P6 scheduleJob bundles.size=${data.size}")
      taskManagerUtils.scheduleJob(context, mTask, data)
    } else {
      Log.w(DIAG, "P6 skip — no bundles after timestamp filter")
    }
  }

  private fun shouldReportDeferredLocations(): Boolean {
    val task = mTask ?: return false
    if (mDeferredLocations.size == 0) {
      return false
    }
    if (!mIsHostPaused) {
      // Don't defer location updates when the activity is in foreground state.
      return true
    }
    val oldestLocation = mLastReportedLocation ?: mDeferredLocations[0]
    val newestLocation = mDeferredLocations[mDeferredLocations.size - 1]
    val options: Arguments = MapHelper(task.options)
    val distance = options.getDouble("deferredUpdatesDistance")
    val interval = options.getLong("deferredUpdatesInterval")
    return newestLocation.time - oldestLocation.time >= interval && mDeferredDistance >= distance
  }

  private fun preparePendingIntent(): PendingIntent {
    return taskManagerUtils.createTaskIntent(context, mTask)
  }

  private fun executeTaskWithLocationBundles(locationBundles: ArrayList<Bundle>, callback: TaskExecutionCallback) {
    if (locationBundles.size > 0 && mTask != null) {
      // P8 — JS task avec locations
      Log.i(DIAG, "P8 executeTask JS=true bundles.size=${locationBundles.size}")
      val data = Bundle()
      data.putParcelableArrayList("locations", locationBundles)
      mTask?.execute(data, null, callback)
    } else {
      // P8 — Finished sans location (smoking gun observé)
      Log.w(
        DIAG,
        "P8 executeTask JS=false FinishedWithoutLocation " +
          "bundles.size=${locationBundles.size} taskNull=${mTask == null}"
      )
      callback.onFinished(null)
    }
  }

  override fun onHostResume() {
    mIsHostPaused = false
    maybeReportDeferredLocations()
  }

  override fun onHostPause() {
    mIsHostPaused = true
  }

  override fun onHostDestroy() {
    mIsHostPaused = true
  } //endregion

  companion object {
    private const val TAG = "LocationTaskConsumer"
    /** Filtre logcat : `adb logcat -s ATMR_LTC_P:I LocationTaskConsumer:W` */
    private const val DIAG = "ATMR_LTC_P"
    private const val FOREGROUND_SERVICE_KEY = "foregroundService"
    private var sLastTimestamp: Long = 0
    fun shouldUseForegroundService(options: Map<String?, Any?>): Boolean {
      return options.containsKey(FOREGROUND_SERVICE_KEY)
    }
  }
}
