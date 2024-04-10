"""Content preprocessors."""
from __future__ import annotations

import copy
import re
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any

from mistletoe import Document  # type: ignore[import]
from mistletoe.block_token import (  # type: ignore[import]
    BlockToken,
    Heading,
    List,
    ListItem,
    Paragraph,
)
from mistletoe.markdown_renderer import MarkdownRenderer  # type: ignore[import]
from mistletoe.span_token import (  # type: ignore[import]
    Image,
    Link,
    RawText,
    SpanToken,
)
from mistletoe.token import Token  # type: ignore[import]

SectionProcessor = Callable[[Any, "DocumentSection"], None]


def section_processor(func: SectionProcessor):
    """Decorator for section processors."""
    setattr(func, "is_section_processor", True)
    return func


def mark_deleted(token: Token | None, deleted: bool = True):
    """Mark a token as deleted."""
    if token is None:
        return
    if not hasattr(token, "meta"):
        setattr(token, "meta", {})
    token.meta["deleted"] = deleted


def is_deleted(token: Token):
    """Check if a token is deleted."""
    return getattr(token, "meta", {}).get("deleted", False)


def gettext(token: Token) -> str:
    """Get the text of a token."""
    with MarkdownRenderer() as renderer:
        text = renderer.render(token)
    return text.strip()


def textmatches(text: str, patterns: Iterable[str]) -> bool:
    """Check if a text matches any of the patterns."""
    for pattern in patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


@dataclass
class DocumentSection:
    """A section of a project description."""

    token: Heading | None
    children: list[Token] = field(default_factory=list)
    subsections: list[DocumentSection] = field(default_factory=list)
    parent: DocumentSection | None = None

    @property
    def title(self):
        """Get the title of the section."""
        if not self.token:
            return "root"
        title = gettext(self.token)
        return title.lstrip("#").strip()

    @property
    def level(self):
        """Get the level of the section."""
        return self.token.level if self.token is not None else 0

    def is_root(self):
        """Check if the section is the root."""
        return self.parent is None

    def add_section(self, heading: Heading):
        """Add a subsection to the section."""
        subsection = DocumentSection(heading, parent=self)
        self.subsections.append(subsection)
        return subsection

    def add_child(self, child: Token):
        """Add a child to the section."""
        self.children.append(child)

    @classmethod
    def from_document(cls, doc: Document):
        """Create a section from a document."""
        root = cls(None)
        stack = [root]
        current_section = lambda: stack[-1]  # noqa: E731
        for token in doc.children:
            if isinstance(token, Heading):
                if token.level == 1:
                    # ignore the root heading
                    current_section().add_child(token)
                    continue
                while current_section().level >= token.level:
                    stack.pop()
                section = current_section().add_section(token)
                stack.append(section)
            else:
                current_section().add_child(token)
        return root

    def traverse(
        self, with_self: bool = True
    ) -> Generator[DocumentSection, None, None]:
        """Traverse the section."""
        if with_self:
            yield self
        for subsection in self.subsections:
            yield from subsection.traverse()


class ProjectDescriptionPreprocessor:
    """Preprocess a project description."""

    def __init__(self, content: str):
        """Initialize the preprocessor."""
        self.content = content

    def _is_redundant_requirement(self, item: ListItem):
        """Check if a requirement is redundant."""
        import re

        content = gettext(item)
        content = re.sub(r"(\d{0,9}[.)]|[+\-*])", " ", content, count=1).strip()
        keywords = (
            "file",
            "push",
            "editor",
            "betty",
            "readme.md",
            "work together",
        )
        for keyword in keywords:
            keyword_found = keyword in content.lower()
            if keyword_found:
                return True
        return False

    def _delete_unused_objectives(self, section: DocumentSection):
        """Delete unused objectives."""
        unused_objectives = (r"copyright",)
        for pattern in unused_objectives:
            if re.match(pattern, section.title, re.IGNORECASE):
                mark_deleted(section.token)

    def _remove_preceeding_paragraph(self, section: DocumentSection):
        """Remove the paragraph from a section."""
        for child in section.children:
            if isinstance(child, Paragraph):
                mark_deleted(child)
                break

    def _is_question_item(self, item: ListItem):
        """Check if an item is a question."""
        content = gettext(item)
        content = re.sub(r"(\d{0,9}[.)]|[+\-*])", " ", content, count=1).strip()
        match = re.match(
            r"^(what|how|why|when|where|which|who|whose|whom)", content, re.I
        )
        return bool(match)

    def _find_question_list(self, learning_objectives):
        """Find the question list within the learning objectives section."""
        for child in learning_objectives.children:
            if isinstance(child, List):
                list = copy.copy(child)
                list.children = []
                for item in child.children:
                    if self._is_question_item(item):
                        list.children.append(item)
                idx = learning_objectives.children.index(child)
                learning_objectives.children[idx] = list
                return list
        return None

    def _merge_question_items(
        self,
        learning_objectives: DocumentSection,
        question_list: List,
        section: DocumentSection,
    ):
        """Merge question items into the question list."""
        for child in self.traverse_tokens(section.children):
            if not isinstance(child, List):
                continue
            if question_list is None:
                question_list = copy.copy(child)
                question_list.children = []
                learning_objectives.children.insert(0, question_list)
            for item in child.children:
                if self._is_question_item(item):
                    question_list.children.append(item)
            mark_deleted(child)

    @section_processor
    def merge_learning_objectives(self, section: DocumentSection):
        """Merge the learning objectives."""
        if section.title != "Learning Objectives":
            return
        learning_objectives = section
        question_list = self._find_question_list(learning_objectives)

        # remove the preceeding paragraph
        self._remove_preceeding_paragraph(learning_objectives)

        for section in learning_objectives.traverse(with_self=False):
            self._delete_unused_objectives(section)
            self._remove_preceeding_paragraph(section)
            if is_deleted(section.token):
                continue
            self._merge_question_items(
                learning_objectives, question_list, section
            )

    @section_processor
    def remove_redundant_requirements(self, section: DocumentSection):
        """Remove redundant requirements."""
        requirement_categories = (
            "General",
            "Shell Scripts",
            "C Scripts",
            "Python Scripts",
            "Python Unit Tests",
        )
        if (
            section.parent
            and (
                section.parent.title == "Requirements"
                and textmatches(section.title, requirement_categories)
            )
            or section.title == "Requirements"
        ):
            for child in section.children:
                if isinstance(child, List):
                    for item in child.children:
                        is_redundant = self._is_redundant_requirement(item)
                        if is_redundant:
                            mark_deleted(item)

        """Remove the copywrite section."""
        if "copyright" in section.title.lower():
            mark_deleted(section.token)

    @section_processor
    def remove_unknown_sections(self, section: DocumentSection):
        """Remove unknown sections."""
        known_sections = (
            "Congrats",
            "General",
            "Requirements",
            "More Info",
            "Learning Objectives",
            "Additional Resources",
            "Resources",
            "Concepts Needed",
            "Background Context",
            ".*Clean Up.*",
            r".+DreamCo",
            "Some things to think about",
            "The Presentation",
            "Must Know",
        )

        known_requirements = (
            "General",
            "Shell Scripts",
            "C Scripts",
            "Python Scripts",
            "Python Unit Tests",
        )

        if not section.parent:
            return
        if section.parent.title == "Requirements" and not textmatches(
            section.title, known_requirements
        ):
            mark_deleted(section.token)

        elif section.parent.is_root() and not textmatches(
            section.title, known_sections
        ):
            mark_deleted(section.token)

    @section_processor
    def remove_unused_info(self, section: DocumentSection):
        """Remove unused info."""
        allowed_sections = (
            r"^Output$",
            r"^Checks$",
            r"^Testing$",
            r"^Compilation$",
            r"^Data Structures$",
            r"^" r"^.+allowed functions.+$",
            # r"^.*class.*$",
        )
        if section.parent and section.parent.title == "More Info":
            for pattern in allowed_sections:
                if re.match(pattern, section.title, re.IGNORECASE):
                    break
            else:
                mark_deleted(section.token)

    @section_processor
    def remove_non_resource_links(self, section: DocumentSection):
        """Remove links from the content."""

        def unwrap_links(children):
            for idx, token in enumerate(children):
                if isinstance(token, Link):
                    children[idx] = RawText(token.title)
                if hasattr(token, "children"):
                    unwrap_links(token.children)

        if section.title in ("Resources", "Additional Resources"):
            return

        unwrap_links([section.token] + section.children)

    @section_processor
    def remove_images(self, section: DocumentSection):
        """Remove images from the content."""
        for token in self.traverse_tokens(section.children):
            if isinstance(token, Image):
                mark_deleted(token)

    @section_processor
    def remove_link_titles(self, section: DocumentSection):
        """Correct links in the content."""
        for token in self.traverse_tokens(section.children):
            if isinstance(token, Link):
                token.title = ""

    @classmethod
    def section_processors(cls) -> Generator[SectionProcessor, None, None]:
        """Get the section processors."""
        for attr in cls.__dict__.values():
            if callable(attr) and getattr(attr, "is_section_processor", False):
                yield attr

    def traverse_tokens(
        self, token: Token | list[Token]
    ) -> Generator[Token, None, None]:
        """Traverse the tokens."""
        if isinstance(token, list):
            for item in token:
                yield from self.traverse_tokens(item)
            return
        yield token
        if hasattr(token, "children"):
            for child in token.children:
                yield from self.traverse_tokens(child)

    def build_block(self, block: BlockToken) -> BlockToken | None:
        """Build a block."""
        children = block.children
        n_children = len(children)
        block.children = []

        if is_deleted(block):
            return None

        for token in children:
            if isinstance(token, BlockToken):
                token = self.build_block(token)
                if token is not None:
                    block.children.append(token)
            elif isinstance(token, SpanToken):
                token = self.build_span(token)
                if token is not None:
                    block.children.append(token)
        if not block.children and n_children:
            return None
        return block

    def build_span(self, span: SpanToken) -> SpanToken | None:
        """Build a span."""
        if is_deleted(span):
            # rendered = gettext(span)
            # print(f"🗑 {rendered}")
            return None
        return span

    def build_section(self, root: DocumentSection) -> list[Token]:
        """Build a document section."""
        tokens: list[Token] = []
        token: Token | None
        children: list[Token] = []
        subsection_tokens: list[Token] = []
        # title = "#" * (root.level or 1) + " " + root.title

        if root.token:
            if is_deleted(root.token):
                # print(title, "🗑")
                return []
            tokens.append(root.token)

        # build direct children
        for token in root.children:
            if isinstance(token, BlockToken):
                token = self.build_block(token)
            elif isinstance(token, SpanToken):
                token = self.build_span(token)
            if token is not None:
                children.append(token)

        # build subsections
        for section in root.subsections:
            subsection_tokens.extend(self.build_section(section))

        if not children and not subsection_tokens:
            # all the children and subsections are empty
            # which means this section would be rendered as empty
            # so we can delete it
            # print(title, "🗑")
            return []
        # print(title, "✅")
        # add the children
        tokens.extend(children)
        # add the subsections
        tokens.extend(subsection_tokens)
        return tokens

    def preprocess(self) -> str:
        """Preprocess the content."""
        # from rich import print

        root = DocumentSection.from_document(Document(self.content))
        processors = tuple(self.section_processors())
        for section in root.traverse():
            for processor in processors:
                processor(self, section)
        # print(root)
        doc = Document([])
        tokens = self.build_section(root)
        doc.children = tokens
        with MarkdownRenderer() as renderer:
            return renderer.render(doc)
