"""JWT authentication helpers for Flask routes."""

import os
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Callable, Optional

import jwt
from flask import g, jsonify, request

from user_manager import User, get_user_manager

JWT_SECRET = os.environ.get("JWT_SECRET", "ann-retrieval-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))


def create_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.username,
        "role": user.role,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def get_token_from_request() -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return request.cookies.get("token")


def resolve_current_user() -> Optional[User]:
    token = get_token_from_request()
    if not token:
        return None
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        return None
    return get_user_manager().get_user(payload.get("sub", ""))


def login_required(view_func: Callable) -> Callable:
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        user = resolve_current_user()
        if user is None:
            return jsonify({"error": "请先登录"}), 401
        g.current_user = user
        return view_func(*args, **kwargs)

    return wrapped


def role_required(*roles: str) -> Callable:
    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            user = resolve_current_user()
            if user is None:
                return jsonify({"error": "请先登录"}), 401
            if user.role not in roles:
                return jsonify({"error": "权限不足"}), 403
            g.current_user = user
            return view_func(*args, **kwargs)

        return wrapped

    return decorator
