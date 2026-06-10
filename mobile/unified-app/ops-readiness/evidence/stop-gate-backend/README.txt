STOP GATE Backend — DELETE /clients/me

Critères PASS:
1. DELETE 200
2. client.is_active=False
3. Relogin refusé
4. Second DELETE 400
5. Ancien JWT → GET /clients/me → 401 ou 403 (si revoke_all_user_tokens)

Preuves: delete-account-test.log, CLOSURE.txt
