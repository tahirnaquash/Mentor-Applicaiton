from security import hash_password

# This will generate the exact string PostgreSQL needs
raw_password = "adminpass123"
hashed = hash_password(raw_password)

print("\n--- COPY THE HASH BELOW ---")
print(hashed)
print("---------------------------\n")