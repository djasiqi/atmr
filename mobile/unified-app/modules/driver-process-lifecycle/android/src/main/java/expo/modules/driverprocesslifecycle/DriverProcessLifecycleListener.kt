package expo.modules.driverprocesslifecycle

import android.app.Application
import expo.modules.core.interfaces.ApplicationLifecycleListener

class DriverProcessLifecycleListener : ApplicationLifecycleListener {
  override fun onCreate(application: Application) {
    DriverStartedActivityCounter.install(application, seedIfEmpty = false)
  }
}
