package expo.modules.driverprocesslifecycle

import android.app.Activity
import android.app.Application
import android.os.Bundle
import java.util.concurrent.CopyOnWriteArrayList

/**
 * DRIVER-RUNTIME-01C-A — compteur process des Activities STARTED.
 * Overlay (GrantPermissions) incrémente sans jamais passer à 0.
 * Home / autre app : toutes STOPPED → 0.
 */
object DriverStartedActivityCounter {
  @Volatile
  var count: Int = 0
    private set

  @Volatile
  private var installed: Boolean = false

  private val listeners = CopyOnWriteArrayList<(Int) -> Unit>()

  @Synchronized
  fun install(application: Application, seedIfEmpty: Boolean) {
    if (installed) {
      if (seedIfEmpty && count == 0) {
        count = 1
        emit()
      }
      return
    }
    installed = true
    application.registerActivityLifecycleCallbacks(
      object : Application.ActivityLifecycleCallbacks {
        override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) = Unit

        override fun onActivityStarted(activity: Activity) {
          synchronized(this@DriverStartedActivityCounter) {
            count += 1
          }
          emit()
        }

        override fun onActivityResumed(activity: Activity) = Unit

        override fun onActivityPaused(activity: Activity) = Unit

        override fun onActivityStopped(activity: Activity) {
          synchronized(this@DriverStartedActivityCounter) {
            count = (count - 1).coerceAtLeast(0)
          }
          emit()
        }

        override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit

        override fun onActivityDestroyed(activity: Activity) = Unit
      }
    )
    if (seedIfEmpty && count == 0) {
      count = 1
      emit()
    }
  }

  fun addListener(listener: (Int) -> Unit) {
    listeners.add(listener)
  }

  fun removeListener(listener: (Int) -> Unit) {
    listeners.remove(listener)
  }

  private fun emit() {
    val current = count
    for (listener in listeners) {
      listener(current)
    }
  }
}
