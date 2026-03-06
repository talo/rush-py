from __future__ import annotations

import html
import re

from docutils import nodes
from docutils.parsers.rst import Directive, directives


class ExessParam(nodes.General, nodes.Element):
    pass


class ExessParamBrief(nodes.TextElement, nodes.Element):
    pass


class ExessParams(Directive):
    has_content = True

    def run(self):
        container = nodes.container(classes=["param-dropdowns"])
        self.state.nested_parse(self.content, self.content_offset, container)
        return [container]


class ExessParamDirective(Directive):
    required_arguments = 1
    has_content = True
    option_spec = {
        "type": directives.unchanged_required,
        "default": directives.unchanged_required,
        "brief": directives.unchanged_required,
        "note": directives.unchanged,
    }

    def run(self):
        name = self.arguments[0].strip()
        param_type = self.options.get("type", "").strip()
        default = self.options.get("default", "").strip()
        brief = self.options.get("brief", "").strip()
        notes, warnings = self._parse_notes(self.options.get("note"))
        brief_nodes, brief_messages = self.state.inline_text(brief, self.lineno)
        has_details = any(line.strip() for line in self.content)

        node = ExessParam()
        node["name"] = name
        node["type"] = param_type
        node["default"] = default
        node["notes"] = notes
        node["has_details"] = has_details

        if brief_nodes:
            brief_node = ExessParamBrief()
            brief_node.extend(brief_nodes)
            node["brief_node"] = brief_node
            node += brief_node

        if has_details:
            self.state.nested_parse(self.content, self.content_offset, node)

        return [node, *brief_messages, *warnings]

    def _parse_notes(self, raw: str | None) -> tuple[list[str], list[nodes.Node]]:
        if not raw:
            return [], []
        parts = re.split(r"[,\s]+", raw.strip())
        notes = [part for part in (p.lower() for p in parts) if part]
        unknown = [note for note in notes if note not in NOTE_META]
        warnings = [
            self.state_machine.reporter.warning(
                f"Unknown exess-param note: {note}",
                line=self.lineno,
            )
            for note in unknown
        ]
        return [note for note in notes if note in NOTE_META], warnings


NOTE_META = {
    "info": ("param-note--info", "Tip"),
    "expert": ("param-note--expert", "Expert"),
    "experimental": ("param-note--experimental", "Experimental"),
    "broken": ("param-note--broken", "Known issues"),
}


def _render_param_line(name: str, param_type: str, default: str) -> str:
    name_html = f"<code>{html.escape(name)}</code>" if name else ""
    type_html = f"<code>{html.escape(param_type)}</code>" if param_type else ""
    default_html = ""
    if default:
        default_text = html.escape(default)
        if _is_code_default(default):
            default_html = f"<code>{default_text}</code>"
        else:
            default_html = f'<span class="param-default-text">{default_text}</span>'
    parts = ['<span class="param-line">', name_html]
    if type_html:
        parts.append(" — ")
        parts.append(type_html)
    if default_html:
        parts.append(" (default: ")
        parts.append(default_html)
        parts.append(")")
    parts.append("</span>")
    return "".join(parts)


def _is_code_default(default: str) -> bool:
    if not default:
        return False
    lower = default.lower()
    if lower in {"required", "unset", "defaults", "depends", "auto", "inferred"}:
        return False
    return not any(ch.isspace() for ch in default)


def _render_notes(notes: list[str]) -> str:
    if not notes:
        return ""
    pieces = ['<span class="param-note-group">']
    for note in notes:
        if note not in NOTE_META:
            continue
        css_class, label = NOTE_META[note]
        pieces.append(
            f'<span class="param-note {css_class}" '
            f'aria-label="{label}" title="{label}"></span>'
        )
    pieces.append("</span>")
    return "".join(pieces) if len(pieces) > 2 else ""


_CHEVRON_SVG = (
    '<span class="sd-summary-state-marker sd-summary-chevron-right">'
    '<svg version="1.1" width="1.5em" height="1.5em" '
    'class="sd-octicon sd-octicon-chevron-right" viewBox="0 0 24 24" '
    'aria-hidden="true"><path d="M8.72 18.78a.75.75 0 0 1 0-1.06L14.44 '
    "12 8.72 6.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-"
    ".018l6.25 6.25a.75.75 0 0 1 0 1.06l-6.25 6.25a.75.75 0 0 1-"
    '1.06 0Z"></path></svg></span>'
)


def visit_exess_param_html(self, node: ExessParam) -> None:
    name = node.get("name", "")
    param_type = node.get("type", "")
    default = node.get("default", "")
    brief_node = node.get("brief_node")
    notes = node.get("notes", [])
    has_details = node.get("has_details", False)
    if brief_node in node.children:
        node.children.remove(brief_node)

    if not has_details:
        self.body.append('<div class="param-row">')
        self.body.append('<div class="param-row-header">')
        self.body.append('<span class="param-row-text">')
        self.body.append(_render_param_line(name, param_type, default))
        if brief_node:
            self.body.append('<span class="param-brief">')
            brief_node.walkabout(self)
            self.body.append("</span>")
        self.body.append("</span>")
        self.body.append(_render_notes(notes))
        self.body.append('<span class="param-row-chevron" aria-hidden="true"></span>')
        self.body.append("</div>")
        return

    self.body.append(
        '<details class="sd-sphinx-override sd-dropdown sd-card param-dropdown">'
    )
    self.body.append('<summary class="sd-summary-title sd-card-header">')
    self.body.append('<span class="sd-summary-text">')
    self.body.append(_render_param_line(name, param_type, default))
    if brief_node:
        self.body.append('<span class="param-brief">')
        brief_node.walkabout(self)
        self.body.append("</span>")
    self.body.append("</span>")
    self.body.append(_render_notes(notes))
    self.body.append(_CHEVRON_SVG)
    self.body.append("</summary>")
    self.body.append('<div class="sd-summary-content sd-card-body param-details">')


def depart_exess_param_html(self, node: ExessParam) -> None:
    if node.get("has_details", False):
        self.body.append("</div></details>")
    else:
        self.body.append("</div>")


def _skip_node(self, node: nodes.Node) -> None:
    raise nodes.SkipNode


def visit_exess_param_brief_html(self, node: ExessParamBrief) -> None:
    pass


def depart_exess_param_brief_html(self, node: ExessParamBrief) -> None:
    pass


def setup(app):
    app.add_node(
        ExessParam,
        html=(visit_exess_param_html, depart_exess_param_html),
        latex=(_skip_node, None),
        text=(_skip_node, None),
        man=(_skip_node, None),
        epub=(_skip_node, None),
    )
    app.add_node(
        ExessParamBrief,
        html=(visit_exess_param_brief_html, depart_exess_param_brief_html),
        latex=(_skip_node, None),
        text=(_skip_node, None),
        man=(_skip_node, None),
        epub=(_skip_node, None),
    )
    app.add_directive("exess-params", ExessParams)
    app.add_directive("exess-param", ExessParamDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
