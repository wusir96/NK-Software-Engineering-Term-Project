"""User account management with JSON file persistence."""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

ROLES = ("user", "researcher", "admin")
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,32}$")

_manager: Optional["UserManager"] = None


@dataclass
class User:
    username: str
    password_hash: str
    role: str = "user"
    created_at: str = ""

    def to_public(self) -> dict:
        return {
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at,
        }


class UserManager:
    DATA_DIR = "data"
    USERS_FILE = os.path.join(DATA_DIR, "users.json")

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self._users: dict[str, User] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.USERS_FILE):
            return
        with open(self.USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("role") == "user":
                item["role"] = "researcher"
            user = User(**item)
            self._users[user.username] = user

    def _save(self):
        with open(self.USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [user.__dict__ for user in self._users.values()],
                f,
                ensure_ascii=False,
                indent=2,
            )

    @property
    def n_users(self) -> int:
        return len(self._users)

    def get_user(self, username: str) -> Optional[User]:
        return self._users.get(username)

    def register(self, username: str, password: str) -> User:
        username = username.strip()
        if not USERNAME_PATTERN.match(username):
            raise ValueError("用户名须为 3-32 位字母、数字或下划线")
        if len(password) < 6:
            raise ValueError("密码至少 6 位")
        if username in self._users:
            raise ValueError("用户名已存在")

        role = "admin" if self.n_users == 0 else "user"
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role=role,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._users[username] = user
        self._save()
        return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self._users.get(username.strip())
        if user is None:
            return None
        if not check_password_hash(user.password_hash, password):
            return None
        return user

    def ensure_default_admin(self, username: str = "admin", password: str = "admin123") -> None:
        """Create default admin when no users exist (development bootstrap)."""
        if self.n_users > 0:
            return
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role="admin",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._users[username] = user
        self._save()
        print(f"Created default admin account: {username} / {password}")

    def list_users(self) -> list:
        return sorted([u.to_public() for u in self._users.values()], key=lambda u: u["created_at"])

    def _count_admins(self) -> int:
        return sum(1 for u in self._users.values() if u.role == "admin")

    def update_role(self, username: str, role: str) -> User:
        if role not in ROLES:
            raise ValueError(f"无效角色，可选: {', '.join(ROLES)}")
        user = self._users.get(username)
        if user is None:
            raise ValueError("用户不存在")
        if user.role == "admin" and role != "admin" and self._count_admins() <= 1:
            raise ValueError("不能修改唯一管理员的角色")
        user.role = role
        self._save()
        return user

    def delete_user(self, username: str, operator: str) -> None:
        if username == operator:
            raise ValueError("不能删除自己的账号")
        user = self._users.get(username)
        if user is None:
            raise ValueError("用户不存在")
        if user.role == "admin" and self._count_admins() <= 1:
            raise ValueError("不能删除唯一的管理员")
        del self._users[username]
        self._save()


def get_user_manager() -> UserManager:
    global _manager
    if _manager is None:
        _manager = UserManager()
    return _manager
