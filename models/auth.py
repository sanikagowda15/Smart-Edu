import os
import re
import pandas as pd

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

USERS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "users.csv"
)


# ============================================================
# LOAD USERS
# ============================================================

def load_users():
    """Load users from CSV file."""

    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(
            columns=[
                "name",
                "email",
                "password"
            ]
        )

    data = pd.read_csv(
        USERS_FILE
    )

    for column in [
        "name",
        "email",
        "password"
    ]:
        if column not in data.columns:
            data[column] = ""

    return data


# ============================================================
# VALIDATE NAME
# ============================================================

def valid_name(name):
    """Validate student name."""

    name = str(name).strip()

    if len(name) < 2:
        return False

    return bool(
        re.fullmatch(
            r"[A-Za-z ]+",
            name
        )
    )


# ============================================================
# VALIDATE EMAIL
# ============================================================

def valid_email(email):
    """Validate email address."""

    email = str(email).strip().lower()

    pattern = (
        r"^[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}$"
    )

    return bool(
        re.fullmatch(
            pattern,
            email
        )
    )


# ============================================================
# VALIDATE PASSWORD
# ============================================================

def valid_password(password):
    """
    Password requirements:

    - Minimum 8 characters
    - One uppercase letter
    - One lowercase letter
    - One number
    - One special character
    """

    password = str(password)

    if len(password) < 8:
        return False

    if not re.search(
        r"[A-Z]",
        password
    ):
        return False

    if not re.search(
        r"[a-z]",
        password
    ):
        return False

    if not re.search(
        r"[0-9]",
        password
    ):
        return False

    if not re.search(
        r"[^A-Za-z0-9]",
        password
    ):
        return False

    return True


# ============================================================
# REGISTRATION ERROR
# ============================================================

def get_registration_error(
    name,
    email,
    password
):
    """Return a useful registration error message."""

    name = str(name).strip()
    email = str(email).strip().lower()
    password = str(password)

    if not valid_name(name):
        return (
            "Name must contain only "
            "letters and spaces."
        )

    if not valid_email(email):
        return (
            "Enter a valid email address. "
            "Example: student@gmail.com"
        )

    if not valid_password(password):
        return (
            "Password must contain at least "
            "8 characters, one uppercase letter, "
            "one lowercase letter, one number "
            "and one special character."
        )

    data = load_users()

    data["email"] = (
        data["email"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if email in data["email"].values:
        return (
            "This email is already registered."
        )

    return None


# ============================================================
# REGISTER USER
# ============================================================

def register_user(
    name,
    email,
    password
):
    """Register a new student."""

    error = get_registration_error(
        name,
        email,
        password
    )

    if error:
        return False

    name = str(name).strip()
    email = str(email).strip().lower()
    password = str(password)

    data = load_users()

    data["email"] = (
        data["email"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Hash password before storing it
    hashed_password = (
        generate_password_hash(
            password
        )
    )

    new_user = pd.DataFrame(
        [{
            "name": name,
            "email": email,
            "password": hashed_password
        }]
    )

    data = pd.concat(
        [
            data,
            new_user
        ],
        ignore_index=True
    )

    data.to_csv(
        USERS_FILE,
        index=False
    )

    return True


# ============================================================
# LOGIN USER
# ============================================================

def login_user(
    email,
    password
):
    """Validate login credentials."""

    email = str(email).strip().lower()
    password = str(password)

    data = load_users()

    if data.empty:
        return None

    data["email"] = (
        data["email"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["password"] = (
        data["password"]
        .astype(str)
        .str.strip()
    )

    user_index = data.index[
        data["email"] == email
    ]

    if len(user_index) == 0:
        return None

    index = user_index[0]

    stored_password = str(
        data.loc[
            index,
            "password"
        ]
    )

    # New hashed passwords
    if stored_password.startswith(
        (
            "pbkdf2:",
            "scrypt:"
        )
    ):
        valid = check_password_hash(
            stored_password,
            password
        )

    # Old plaintext passwords
    else:
        valid = (
            stored_password == password
        )

        # Automatically upgrade old
        # plaintext password to hash
        if valid:
            data.loc[
                index,
                "password"
            ] = generate_password_hash(
                password
            )

            data.to_csv(
                USERS_FILE,
                index=False
            )

    if valid:
        return data.loc[
            index,
            "name"
        ]

    return None