"""Migrate data from SQLAlchemy to MongoDB."""

from alexis import logging
from alexis.app import create_app
from alexis.models import Chat, MChat, MThread, MUser, Thread, User


def migrate_users():
    """Migrate users to MongoDB."""
    all_users = User.query.all()
    count = 0
    logging.info(f"Migrating {len(all_users)} users to MongoDB.")
    for user in all_users:
        logging.info(f"Migrating user: {user.uid[:8]}")
        if MUser.objects.filter(email=user.email).first():
            logging.warning("User already exists in MongoDB.")
            continue
        MUser.create(
            id=user.id,
            kinde_user=user.kinde_user,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            picture=user.picture,
        )
        count += 1
    logging.info(f"Migrated {count} users to MongoDB.")


def migrate_threads():
    """Migrate threads to MongoDB."""
    all_threads = Thread.query.all()
    count = 0
    logging.info(f"Migrating {len(all_threads)} threads to MongoDB.")
    for thread in all_threads:
        logging.info(f"Migrating thread: {thread.uid[:8]}")
        if MThread.objects.filter(id=thread.id).first():
            logging.warning("Thread already exists in MongoDB.")
            continue
        user = MUser.objects.filter(id=thread.user_id).first()
        if not user:
            logging.warning("User %r not found in MongoDB.", thread.user_id)
            continue
        MThread.create(
            id=thread.id,
            user=user,
            title=thread.title,
            project=thread.project,
            closed=thread.closed,
        )
        count += 1
    logging.info(f"Migrated {count} threads to MongoDB.")


def migrate_chats():
    """Migrate chats to MongoDB."""
    all_chats = Chat.query.all()
    count = 0
    logging.info(f"Migrating {len(all_chats)} chats to MongoDB.")
    for chat in all_chats:
        logging.info(f"Migrating chat: {chat.uid[:8]}")
        if MChat.objects.filter(id=chat.id).first():
            logging.warning("Chat already exists in MongoDB.")
            continue
        thread = MThread.objects.filter(id=chat.thread_id).first()
        if not thread:
            logging.warning("Thread %r not found in MongoDB.", chat.thread_id)
            continue
        previous_chat = MChat.objects.filter(id=chat.previous_chat_id).first()
        if not previous_chat and chat.previous_chat_id is not None:
            logging.warning(
                "Previous chat %r not found in MongoDB.", chat.previous_chat_id
            )
            continue
        chat = thread.add_chat(
            id=chat.id,
            chat_type=chat.chat_type,
            content=chat.content,
            cost=chat.cost,
            sent_time=chat.sent_time,
            previous_chat=previous_chat,
        )
        count += 1
    logging.info(f"Migrated {count} chats to MongoDB.")


def migrate_to_mongo():
    """Migrate data to MongoDB."""
    logging.info("Starting migration.")
    create_app()
    migrate_users()
    migrate_threads()
    migrate_chats()
    logging.info("Migration complete.")


if __name__ == "__main__":
    migrate_to_mongo()
