from app import create_app
app = create_app()
print("=== Routes SMTP ===")
smtp_routes = [str(rule) for rule in app.url_map.iter_rules() if "smtp" in str(rule)]
if smtp_routes:
    for route in smtp_routes:
        print(route)
else:
    print("❌ Aucune route SMTP trouvée!")
