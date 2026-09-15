from accounts.auth import generate_session_token, hash_password, verify_password

def test_hash_password_produces_different_salts_each_time():
    salt1, hash1 = hash_password("hunter2222")
    salt2, hash2 = hash_password("hunter2222")
    assert salt1 != salt2
    assert hash1 != hash2

def test_verify_password_accepts_correct_password():
    salt, hash_hex = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", salt, hash_hex) is True

def test_verify_password_rejects_wrong_password():
    salt, hash_hex = hash_password("correct horse battery staple")
    assert verify_password("wrong password", salt, hash_hex) is False

def test_generate_session_token_is_unique_and_reasonably_long():
    tokens = {generate_session_token() for _ in range(20)}
    assert len(tokens) == 20
    assert all(len(t) > 20 for t in tokens)
