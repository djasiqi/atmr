package expo.modules.driverprocesslifecycle

import android.app.Application
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class DriverProcessLifecycleModule : Module() {
  private val countListener: (Int) -> Unit = { count ->
    sendEvent(
      "onStartedActivityCountChanged",
      mapOf("count" to count)
    )
  }

  override fun definition() = ModuleDefinition {
    Name("DriverProcessLifecycle")

    Events("onStartedActivityCountChanged")

    OnCreate {
      val application = appContext.reactContext?.applicationContext as? Application
      if (application != null) {
        DriverStartedActivityCounter.install(application, seedIfEmpty = true)
      }
    }

    OnStartObserving {
      DriverStartedActivityCounter.addListener(countListener)
    }

    OnStopObserving {
      DriverStartedActivityCounter.removeListener(countListener)
    }

    Function("getStartedActivityCount") {
      DriverStartedActivityCounter.count
    }
  }
}
