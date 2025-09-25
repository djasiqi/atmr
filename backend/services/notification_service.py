import requests
from flask_socketio import emit
from app import socketio

# 🔔 1. Notification PUSH Expo
def send_push_message(token, title, body):
    message = {
        'to': token,
        'sound': 'default',
        'title': title,
        'body': body,
    }
    response = requests.post('https://exp.host/--/api/v2/push/send', json=message)
    return response.json()


# 🟢 2. Notification WebSocket – nouvelle course
def notify_driver_new_booking(driver_id, booking):
    room = f"driver_{driver_id}"
    socketio.emit("new_booking", booking.to_dict(), to=room)


# 🔄 3. Notification WebSocket – mise à jour de mission
def notify_booking_update(driver_id, booking):
    room = f"driver_{driver_id}"
    socketio.emit("booking_updated", booking.to_dict(), to=room)


# ❌ 4. Notification WebSocket – annulation de mission
def notify_booking_cancelled(driver_id, booking_id):
    room = f"driver_{driver_id}"
    socketio.emit("booking_cancelled", {"id": booking_id}, to=room)

def notify_booking_assigned(booking):
    """
    Notification unifiée quand une réservation est assignée.
    - Émet un événement SocketIO côté entreprise
    - (Optionnel) ici tu peux ajouter SMS/Email/Push si nécessaire
    """
    try:
        from services.socketio_service import emit_company_event
        payload = {
            "booking_id": booking.id,
            "driver_id": booking.driver_id,
            "status": str(booking.status) if hasattr(booking, "status") else None,
        }
        emit_company_event(booking.company_id, "booking_assigned", payload)
    except Exception as e:
        # On logge, mais on ne bloque pas le flux d'assignation
        try:
            from ext import app_logger
            app_logger.error(f"[notify_booking_assigned] emit failed: {e}")
        except Exception:
            pass

def notify_dispatch_run_completed(company_id, dispatch_run_id, assignments_count, date_str=None):
    """
    Notification unifiée quand un run de dispatch est terminé.
    
    Args:
        company_id: ID de l'entreprise
        dispatch_run_id: ID du run de dispatch
        assignments_count: Nombre d'assignations créées
        date_str: Date du run au format YYYY-MM-DD (optionnel)
    """
    try:
        from services.socketio_service import emit_company_event, emit_date_event
        
        # If date_str is not provided, try to get it from the DispatchRun model
        if not date_str:
            try:
                from models import DispatchRun
                dispatch_run = DispatchRun.query.get(dispatch_run_id)
                if dispatch_run and dispatch_run.day:
                    # Ensure we get a string representation of the date
                    date_str = dispatch_run.day.isoformat() if hasattr(dispatch_run.day, 'isoformat') else str(dispatch_run.day)
                    # Log that we found the date
                    from ext import app_logger
                    app_logger.info(f"[notify_dispatch_run_completed] Retrieved date_str={date_str} from dispatch_run_id={dispatch_run_id}")
            except Exception as e:
                from ext import app_logger
                app_logger.warning(f"[notify_dispatch_run_completed] Failed to get day_str from DispatchRun: {e}")
        
        payload = {
            "dispatch_run_id": dispatch_run_id,
            "assignments_count": assignments_count,
            "date": date_str,
        }
        
        # Log the payload for debugging
        from ext import app_logger
        app_logger.info(f"[notify_dispatch_run_completed] Emitting event with payload: {payload}")
        
        # Émettre l'événement vers la room de l'entreprise
        emit_company_event(company_id, "dispatch_run_completed", payload)
        
        # Si on a la date, émettre aussi vers la room de date
        if date_str:
            emit_date_event(date_str, "dispatch_run_completed", payload)
    except Exception as e:
        # On logge, mais on ne bloque pas le flux
        try:
            from ext import app_logger
            app_logger.error(f"[notify_dispatch_run_completed] emit failed: {e}")
        except Exception:
            pass