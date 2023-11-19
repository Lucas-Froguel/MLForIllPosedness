from dataclasses import dataclass



@dataclass
class UserNotFound:
    _message: str = ("Could not find any user with the provide data.")


@dataclass
class UserAlreadyPresent:
    _message: str = ("User already has posted cities.")


@dataclass
class DatabaseError:
    _message: str = ("Something went wrong while connecting to the database.")
    details: str = ""


@dataclass
class ItemNotFound:
    _message: str = ("Item not found.")
    details: str = ""
