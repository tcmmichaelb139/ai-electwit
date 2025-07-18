import os
import random
import logging
import hashlib
from typing import List

from electwit.utils import get_closest_response

SEED = os.getenv("SEED")
random.seed(SEED)

logger = logging.getLogger(__name__)


class Comment:
    """
    Represents a comment on a post.
    Each comment can have replies and likes.
    """

    def __init__(
        self, id: str, name: str, day: int, hour: int, content: str, depth: int = 0
    ):
        self.id = id
        self.name = name
        self.day = day
        self.hour = hour
        self.content = content[:280]
        if len(content) > 280:
            self.content += "..."
        self.replies: List[Comment] = []
        self.likes = 0
        self.depth = depth

    def add_comment(self, comment: "Comment") -> None:
        """Adds a reply to the comment."""
        comment.depth = self.depth + 1

        self.replies.append(comment)

    def _export_to_json(self) -> dict:
        """
        Exports the comment to a JSON-compatible dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "day": self.day,
            "hour": self.hour,
            "content": self.content,
            "replies": [reply._export_to_json() for reply in self.replies],
            "likes": self.likes,
        }


class Post:
    """
    Represents a post on the platform.
    Each post can have replies and likes.
    """

    def __init__(self, id: str, name: str, day: int, hour: int, content: str):
        self.id = id
        self.name = name
        self.day = day
        self.hour = hour
        self.content = content[:280]
        if len(content) > 280:
            self.content += "..."
        self.replies: List[Comment] = []
        self.likes = 0

    def add_comment(self, comment: Comment) -> None:
        """Adds a comment to the post."""
        self.replies.append(comment)

    def _export_to_json(self) -> dict:
        """
        Exports the post to a JSON-compatible dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "day": self.day,
            "hour": self.hour,
            "content": self.content,
            "replies": [reply._export_to_json() for reply in self.replies],
            "likes": self.likes,
        }


class Platform:
    """
    Represents the platform where the election simulation takes place.
    This class is responsible for managing the agents and their interactions.
    """

    def __init__(self):
        self.platform: List[Post] = []
        self.ids = []

    def apply_actions(self, name: str, day: int, hour: int, actions: List[dict]):
        """
        Applies a list of actions to the platform.
        """

        for action in actions:
            action_action = get_closest_response(
                testing_text=action["action"], responses=["POST", "REPLY", "LIKE"]
            )
            if action_action == "POST":
                new_id = self._get_new_id(action)
                if not new_id:
                    continue
                self.platform.append(
                    Post(
                        id=new_id,
                        name=name,
                        day=day,
                        hour=hour,
                        content=action["content"],
                    )
                )
            elif action_action == "REPLY":
                replying_to = self._find_thing_by_id(action["id"])
                if not replying_to:
                    logger.warning(
                        f"Could not find post/comment with id {action['id']} to reply to. Action skipped."
                    )
                    continue
                new_id = self._get_new_id(action)
                if not new_id:
                    continue
                replying_to.add_comment(
                    Comment(
                        id=new_id,
                        name=name,
                        day=day,
                        hour=hour,
                        content=action["content"],
                    )
                )
            elif action_action == "LIKE":
                liking = self._find_thing_by_id(action["id"])
                if not liking:
                    logger.warning(
                        f"Could not find post/comment with id {action['id']} to like. Action skipped."
                    )
                    continue
                liking.likes += 1
            else:
                logger.error(f"Unknown action: {action['action']}. Action skipped.")

    def _get_new_id(self, action: dict, depth: int = 0) -> str:
        """
        Generates a new unique ID for a post.
        """
        if depth > 5:
            logger.error(
                "Maximum recursion depth reached while generating a new ID. Returning empty string."
            )
            return ""

        hashstr = str(action) + str(len(self.ids)) + str(depth)

        hashid = hashlib.sha1(hashstr.encode("utf-8")).hexdigest()[:10]

        if hashid in self.ids:
            logger.warning(f"ID {hashid} already exists. Generating a new one.")
            return self._get_new_id(action, depth=depth + 1)

        self.ids.append(hashid)

        return hashid

    def _recursive_find(
        self, id: str, items: List[Post | Comment]
    ) -> Post | Comment | None:
        """
        Recursively finds a post or comment by its ID.
        """

        for item in items:
            if item.id == id:
                return item
            found = self._recursive_find(id, item.replies)
            if found:
                return found

        return None

    def _find_thing_by_id(self, id_: str) -> Post | Comment | None:
        """
        Finds both a post or a comment by its ID.
        """

        if id_ not in self.ids:
            return None

        result = self._recursive_find(id_, self.platform)

        return result

    def get_feed(self, limit: int = 10, rand: bool = True) -> List[Post]:
        """
        Returns the most recent posts from the platform.

        Args:
        - limit (int): The maximum number of posts to return.
        """

        feed = self.platform[-limit:].copy()

        if rand:
            random.shuffle(feed)

        return feed

    def _comment_thread_string(self, comment: Comment, indent: str = "  ") -> str:
        """
        Recursively formats a comment and its replies into a string.

        Args:
        - comment (Comment): The comment to format.
        - indent (str): The indentation string for nested replies.
        """
        result = f"{indent}Reply from {comment.name} (ID: {comment.id}) (Day: {comment.day} Hour: {comment.hour}): {comment.content}\n(Likes: {comment.likes})\n"
        if comment.depth <= 3:  # Limit depth to 3 for readability
            for reply in comment.replies:
                result += self._comment_thread_string(reply, indent + "  ")
        return result

    def get_feed_as_string(self, limit: int = 10, rand: bool = True) -> str:
        """
        Returns the most recent posts as a formatted string.

        Args:
        - limit (int): The maximum number of posts to return.
        """
        feed = self.get_feed(limit, rand)

        result = ""

        for post in feed:
            result += f"Post from {post.name} (ID: {post.id}) (Day: {post.day} Hour: {post.hour}):\n{post.content}\n(Likes: {post.likes})\n"
            for comment in post.replies:
                result += self._comment_thread_string(comment, indent="  ")

        if not result:
            return "Feed is empty."

        return result.strip()

    def __str__(self):
        """
        Returns a string representation of the platform.
        """

        return (
            self.get_feed_as_string(limit=len(self.platform), rand=False)
            or "No posts available."
        )

    def _export_to_json(self) -> dict:
        """
        Exports the platform to a JSON-compatible dictionary.
        """
        return {
            "platform": [post._export_to_json() for post in self.platform],
            "ids": self.ids,
        }
