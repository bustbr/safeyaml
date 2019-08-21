#!/usr/bin/env python3

import argparse
import io
import json
import re
import sys
from collections import OrderedDict

quote_chars = "\"'"

whitespace_h_chars = " ", "\t"
whitespace_v_chars = "\r", "\n"
whitespace_chars = whitespace_h_chars + whitespace_v_chars
whitespace = re.compile(r"(?:\ |\t|\r|\n)+")

comment = re.compile(r"(#[^\r\n]*(?:\r?\n|$))+")

int_b10 = re.compile(r"\d[\d]*")
flt_b10 = re.compile(r"\.[\d]+")
exp_b10 = re.compile(r"[eE](?:\+|-)?[\d+]")

string_esc = r"\\(?:['\"\\/bfnrt]|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})"
string_dq = re.compile(r'"(?:[^"\\\n\x00-\x1F\uD800-\uDFFF]|{})*"'.format(string_esc))
string_sq = re.compile(r"'(?:[^'\n\x00-\x1F\uD800-\uDFFF]|{})*'".format(string_esc))

identifier = re.compile(r"(?!\d)[\w.$]+")
barewords = re.compile(
    r"(?!\d)(?:(?![\r\n#$@%`,:\"\|'\[\]\{\}\&\*\?\<\>]).|:[^\r\n\s])*"
)

key_name = re.compile(
    "(?:{}|{}|{})".format(string_dq.pattern, string_sq.pattern, identifier.pattern)
)

str_escapes = {
    "b": "\b",
    "n": "\n",
    "f": "\f",
    "r": "\r",
    "t": "\t",
    "/": "/",
    '"': '"',
    "'": "'",
    "\\": "\\",
}

builtin_names = {"null": None, "true": True, "false": False}

reserved_names = set("y|n|yes|no|on|off".split("|"))

newlines = re.compile(r"\r?\n")  # Todo: unicode


def get_position(buf, pos):
    "Given a new offset, find the next position"
    line = 1
    line_off = 0
    for match in newlines.finditer(buf, 0, pos):
        line += 1
        line_off = match.end()

    col = (pos - line_off) + 1
    return line, col


class Options:
    def __init__(self, force_string_keys=False, force_commas=False):
        self.force_string_keys = force_string_keys
        self.force_commas = force_commas


class ParserErr(Exception):
    def name(self):
        return self.__class__.__name__

    def explain(self):
        return self.reason

    def __init__(self, buf, pos, reason=None):
        self.buf = buf
        self.pos = pos
        if reason is None:
            nl = buf.rfind(" ", pos - 10, pos)
            if nl < 0:
                nl = pos - 5
            reason = "Unknown Character {} (context: {})".format(
                repr(buf[pos]), repr(buf[pos - 10 : pos + 5])
            )
        self.reason = reason
        Exception.__init__(self, "{} (at pos={})".format(reason, pos))


class SemanticErr(ParserErr):
    pass


class DuplicateKey(SemanticErr):
    pass


class ReservedKey(SemanticErr):
    pass


class SyntaxErr(ParserErr):
    pass


class BadIndent(SyntaxErr):
    pass


class BadKey(SyntaxErr):
    pass


class Bareword(ParserErr):
    pass


class BadString(SyntaxErr):
    pass


class BadNumber(SyntaxErr):
    pass


class NoRootObject(SyntaxErr):
    pass


class ObjectIndentationErr(SyntaxErr):
    pass


class TrailingContent(SyntaxErr):
    pass


class UnsupportedYAML(ParserErr):
    pass


class UnsupportedEscape(ParserErr):
    pass


def parse(buf, output=None, options=None):
    if not buf:
        raise NoRootObject(buf, 0, "Empty Document")

    output = output or io.StringIO()
    options = options or Options()
    pos = 1 if buf.startswith("\uFEFF") else 0

    out = []
    while pos != len(buf):
        obj, pos = parse_document(buf, pos, output, options)
        out.append(obj)

        if buf[pos : pos + 3] == "---":
            output.write(buf[pos : pos + 3])
            pos += 3
        elif pos < len(buf):
            raise TrailingContent(
                buf, pos, "Trailing content: {}".format(repr(buf[pos : pos + 10]))
            )

    return out


def parse_document(buf, pos, output, options):
    obj, pos = parse_structure(buf, pos, output, options, at_root=True)

    start = pos
    m = whitespace.match(buf, pos)
    while m:
        pos = m.end()
        m = comment.match(buf, pos)
        if m:
            pos = m.end()
            m = whitespace.match(buf, pos)
    output.write(buf[start:pos])
    return obj, pos


def peek_line(buf, pos):
    start = pos
    while pos < len(buf):
        peek = buf[pos]
        if peek in ("\r", "\n"):
            break
        pos += 1
    return buf[start:pos]


def skip_to_eol(buf, pos):
    peek = ""
    while pos < len(buf):
        peek = buf[pos]
        if peek in whitespace_v_chars:
            break
        else:
            pos += 1
    return pos, peek


def skip_h_whitespaces(buf, pos):
    peek = ""
    while pos < len(buf):
        peek = buf[pos]
        if peek in whitespace_h_chars:
            pos += 1
        else:
            break
    return pos, peek


def move_to_next(buf, pos):
    line_pos = pos
    next_line = False
    while pos < len(buf):
        peek = buf[pos]

        if peek == " ":
            pos += 1
        elif peek == "\n" or peek == "\r":
            pos += 1
            line_pos = pos
            next_line = True
        elif peek == "#":
            next_line = True
            while pos < len(buf):
                pos += 1
                if buf[pos] == "\r" or buf[pos] == "\n":
                    line_pos = pos
                    next_line = True
                    break
        else:
            break
    return pos, pos - line_pos, next_line


def skip_whitespace(buf, pos, output=None):
    m = whitespace.match(buf, pos)
    while m:
        if output:
            output.write(buf[pos : m.end()])
        pos = m.end()
        m = comment.match(buf, pos)
        if m:
            if output:
                output.write(buf[pos : m.end()])
            pos = m.end()
            m = whitespace.match(buf, pos)
    return pos


def parse_structure(buf, pos, output, options, indent=0, at_root=False):
    while True:
        start = pos
        pos, my_indent, next_line = move_to_next(buf, pos)

        if my_indent < indent:
            raise BadIndent(
                buf,
                pos,
                "The parser has gotten terribly confused, I'm sorry. Try re-indenting",
            )

        output.write(buf[start:pos])
        peek = buf[pos]

        if peek in ("*", "&", "?", "|", "<", ">", "%", "@"):
            raise UnsupportedYAML(
                buf,
                pos,
                "I found a {} outside of quotes. It's too special to let pass. Anchors, References, and other directives are not valid SafeYAML, Sorry.".format(
                    peek
                ),
            )

        if peek == "-" and buf[pos : pos + 3] == "---":
            output.write(buf[pos : pos + 3])
            pos += 3
            continue
        break

    if peek == "-":
        return parse_indented_list(buf, pos, output, options, my_indent)

    m = key_name.match(buf, pos)

    if peek in quote_chars or m:
        return parse_indented_map(buf, pos, output, options, my_indent, at_root)

    if peek == "{":
        if at_root:
            return parse_map(buf, pos, output, options)
        else:
            raise BadIndent(
                buf,
                pos,
                "Expected an indented object or indented list, but found {} on next line",
            )
    if peek == "[":
        if at_root:
            return parse_list(buf, pos, output, options)
        else:
            raise BadIndent(
                buf,
                pos,
                "Expected an indented object or indented list, but found [] on next line",
            )

    if peek in "+-0123456789":
        if at_root:
            raise NoRootObject(
                buf,
                pos,
                "No root object found: expected object or list, found start of number",
            )
        else:
            raise BadIndent(
                buf,
                pos,
                "Expected an indented object or indented list, but found start of number on next line.",
            )

    raise SyntaxErr(buf, pos, "The parser has become terribly confused, I'm sorry")


def parse_indented_list(buf, pos, output, options, my_indent):
    out = []
    while pos < len(buf):
        if buf[pos] != "-":
            break
        output.write("-")
        pos += 1
        if buf[pos] not in (" ", "\r", "\n"):
            raise BadKey(
                buf,
                pos,
                "For indented lists i.e '- foo', the '-'  must be followed by ' ', or '\n', not: {}".format(
                    buf[pos - 1 : pos + 1]
                ),
            )

        new_pos, new_indent, next_line = move_to_next(buf, pos)
        if next_line and new_indent <= my_indent:
            raise BadIndent(
                buf,
                new_pos,
                "Expecting a list item, but the next line isn't indented enough",
            )

        if not next_line:
            output.write(buf[pos:new_pos])
            line = peek_line(buf, pos)
            if ": " in line:
                new_indent = my_indent + 1 + (new_pos - pos)
                obj, pos = parse_indented_map(
                    buf, new_pos, output, options, new_indent, at_root=False
                )
            else:
                obj, pos = parse_value(buf, new_pos, output, options)

        else:
            obj, pos = parse_structure(buf, pos, output, options, indent=my_indent)

        out.append(obj)

        new_pos, new_indent, next_line = move_to_next(buf, pos)
        if next_line and new_indent == my_indent and buf[new_pos : new_pos + 1] == "-":
            output.write(buf[pos:new_pos])
            pos = new_pos
            continue

        break

    return out, pos


def parse_indented_map(buf, pos, output, options, my_indent, at_root):
    out = OrderedDict()

    while pos < len(buf):
        m = key_name.match(buf, pos)
        if not m:
            break

        name, pos, is_bare = parse_key(buf, pos, output, options)
        if name in out:
            raise DuplicateKey(
                buf,
                pos,
                "Can't have duplicate keys: {} is defined twice.".format(repr(name)),
            )

        peek = buf[pos] if pos < len(buf) else ""
        if peek != ":":
            if peek in whitespace_h_chars:
                pos = skip_whitespace(buf, pos)
            elif is_bare or not at_root:
                raise BadKey(
                    buf,
                    pos,
                    "Expected 'key:', but didn't find a ':', found {}".format(
                        repr(buf[pos:])
                    ),
                )
            else:
                raise NoRootObject(
                    buf,
                    pos,
                    "Expected 'key:', but didn't find a ':', found a string {}. Note that strings must be inside a containing object or list, and cannot be root element".format(
                        repr(buf[pos:])
                    ),
                )

        output.write(":")
        pos += 1
        if buf[pos] not in whitespace_v_chars:
            output.write(" ")

        new_pos, new_indent, next_line = move_to_next(buf, pos)
        if next_line and new_indent < my_indent:
            raise BadIndent(
                buf,
                new_pos,
                "Missing value. Found a key, but the line afterwards isn't indented enough to count.",
            )

        if not next_line:
            obj, pos = parse_value(buf, new_pos, output, options)
        else:
            output.write(buf[pos : new_pos - new_indent])
            obj, pos = parse_structure(
                buf, new_pos - new_indent, output, options, indent=my_indent
            )

        # dupe check
        out[name] = obj

        pos, peek = skip_h_whitespaces(buf, pos)
        if peek == "#":
            output.write(" #")
            pos, peek = skip_h_whitespaces(buf, pos + 1)
            new_pos, peek = skip_to_eol(buf, pos)
            comment = buf[pos:new_pos].strip()
            if comment:
                output.write(" " + comment)
            pos = new_pos

        new_pos, new_indent, next_line = move_to_next(buf, pos)
        if not next_line or new_indent != my_indent:
            break
        else:
            output.write(buf[pos:new_pos])
            pos = new_pos

    return out, pos


def parse_value(buf, pos, output, options=None):
    pos = skip_whitespace(buf, pos, output)

    peek = buf[pos]

    if peek in ("*", "&", "?", "|", "<", ">", "%", "@"):
        raise UnsupportedYAML(
            buf,
            pos,
            "I found a {} outside of quotes. It's too special to let pass. Anchors, References, and other directives are not valid SafeYAML, Sorry.".format(
                peek
            ),
        )

    if peek == "-" and buf[pos : pos + 3] == "---":
        raise UnsupportedYAML(
            buf,
            pos,
            "A SafeYAML document is a single document, '---' separators are unsupported",
        )

    if peek == "{":
        return parse_map(buf, pos, output, options)
    elif peek == "[":
        return parse_list(buf, pos, output, options)
    elif peek == "'" or peek == '"':
        return parse_string(buf, pos, output, options)
    elif peek in "-+0123456789":
        return parse_number(buf, pos, output, options)
    else:
        return parse_bareword(buf, pos, output, options)

    # raise ParserErr(buf, pos, "Bug in parser, sorry")


def parse_map(buf, pos, output, options):
    output.write("{")
    out = OrderedDict()

    pos += 1
    pos = skip_whitespace(buf, pos)

    comma = None

    while buf[pos] != "}":

        key, new_pos, is_bare = parse_key(buf, pos, output, options)

        if key in out:
            raise DuplicateKey(buf, pos, "duplicate key: {}, {}".format(key, out))

        pos = skip_whitespace(buf, new_pos)

        peek = buf[pos]

        if peek == ":":
            output.write(":")
            pos += 1
        else:
            raise BadKey(
                buf,
                pos,
                "Expected a ':', when parsing a key: value pair but found {}".format(
                    repr(peek)
                ),
            )

        pos = skip_whitespace(buf, pos)
        output.write(" ")

        item, pos = parse_value(buf, pos, output, options)

        # dupe check
        out[key] = item

        pos = skip_whitespace(buf, pos)

        peek = buf[pos]
        comma = False
        if peek == ",":
            pos += 1
            output.write(",")
            comma = True
        elif peek != "}":
            raise SyntaxErr(
                buf,
                pos,
                "Expecting a ',', or a '{}' but found {}".format("}", repr(peek)),
            )

        peek = buf[pos]

        if peek not in whitespace_v_chars + ("}",):
            output.write(" ")

        pos = skip_whitespace(buf, pos)

    if options.force_commas:
        if out and comma == False:
            output.write(",")
    output.write("}")
    return out, pos + 1


def parse_key(buf, pos, output, options):
    quote_char = buf[pos] if buf[pos] in quote_chars else ""
    is_bare = not quote_char
    if is_bare:
        name, pos = parse_bare_key(buf, pos, output, options)
    else:
        try:
            name, pos = parse_bare_key(buf, pos + 1, output, options)
            pos += 1
        except (BadKey, ReservedKey):
            name, pos = parse_string(buf, pos, output, options)
    pos = skip_whitespace(buf, pos)
    return name, pos, is_bare


def parse_bare_key(buf, pos, output, options):
    start_pos = end_pos = pos
    while True:
        m = identifier.match(buf, pos)
        if not m:
            break
        end_pos = m.end()
        pos = skip_whitespace(buf, end_pos)

    item = buf[start_pos:end_pos]
    if not item:
        raise BadKey(buf, start_pos, "Found no key.")
    name = item.lower()

    if (
        options.force_string_keys
        or name in builtin_names
        or name in reserved_names
        or " " in name
    ):
        item = '"{}"'.format(item)

    output.write(item)
    return name, end_pos


def parse_list(buf, pos, output, options):
    output.write("[")
    out = []

    pos += 1

    pos = skip_whitespace(buf, pos, output)
    comma = None

    while buf[pos] != "]":
        item, pos = parse_value(buf, pos, output, options)
        out.append(item)

        pos = skip_whitespace(buf, pos, output)

        peek = buf[pos]
        comma = False
        if peek == ",":
            output.write(",")
            comma = True
            pos += 1
            pos = skip_whitespace(buf, pos, output)
        elif peek != "]":
            raise SyntaxErr(
                buf,
                pos,
                "Inside a [], Expecting a ',', or a ']' but found {}".format(
                    repr(peek)
                ),
            )
    if options.force_commas:
        if out and comma == False:
            output.write(",")
    output.write("]")
    pos += 1

    return out, pos


def parse_string(buf, pos, output, options):
    s = io.StringIO()
    is_sq = buf[pos] == "'"

    # validate string
    if is_sq:
        start = end = pos
        while True:
            m = string_sq.match(buf, start)
            if not m:
                break
            start = end = m.end()
        if pos == end:
            raise BadString(buf, pos, "Invalid single quoted string")
        output.write(buf[pos:end])
    else:
        m = string_dq.match(buf, pos)
        if m:
            end = m.end()
            output.write(buf[pos:end])
        else:
            raise BadString(buf, pos, "Invalid double quoted string")

    if is_sq:
        out = buf[pos + 1 : end - 1]
        return out.replace("''", "'"), end

    lo = pos + 1  # skip quotes
    while lo < end - 1:
        hi = buf.find("\\", lo, end)
        if hi == -1:
            s.write(buf[lo : end - 1])  # skip quote
            break

        s.write(buf[lo:hi])

        esc = buf[hi + 1]
        if esc in str_escapes:
            s.write(str_escapes[esc])
            lo = hi + 2
        elif esc == "x":
            n = int(buf[hi + 2 : hi + 4], 16)
            s.write(chr(n))
            lo = hi + 4
        elif esc == "u":
            n = int(buf[hi + 2 : hi + 6], 16)
            if 0xD800 <= n <= 0xDFFF:
                raise BadString(buf, hi, "string cannot have surrogate pairs")
            s.write(chr(n))
            lo = hi + 6
        elif esc == "U":
            n = int(buf[hi + 2 : hi + 10], 16)
            if 0xD800 <= n <= 0xDFFF:
                raise BadString(buf, hi, "string cannot have surrogate pairs")
            s.write(chr(n))
            lo = hi + 10
        else:
            raise UnsupportedEscape(
                buf, hi, "Unkown escape character {}".format(repr(esc))
            )

    out = s.getvalue()

    # XXX output.write string.escape

    return out, end


def parse_number(buf, pos, output, options):
    flt_end = None
    exp_end = None

    sign = +1

    start = pos

    if buf[pos] in "+-":
        if buf[pos] == "-":
            sign = -1
        pos += 1
    peek = buf[pos]

    leading_zero = peek == "0"
    m = int_b10.match(buf, pos)
    if m:
        int_end = m.end()
        end = int_end
    else:
        raise BadNumber(buf, pos, "Invalid number")

    t = flt_b10.match(buf, end)
    if t:
        flt_end = t.end()
        end = flt_end
    e = exp_b10.match(buf, end)
    if e:
        exp_end = e.end()
        end = exp_end

    if flt_end or exp_end:
        out = sign * float(buf[pos:end])
    else:
        out = sign * int(buf[pos:end])
        if leading_zero and out != 0:
            raise BadNumber(buf, pos, "Can't have leading zeros on non-zero integers")

    output.write(buf[start:end])

    return out, end


def parse_bareword(buf, pos, output, options):
    m = identifier.match(buf, pos)
    item = None
    if m:
        end = m.end()
        item = buf[pos:end]
        name = item.lower()

        if name in builtin_names:
            out = builtin_names[name]
            output.write(name)
            return out, m.end()

    m = barewords.match(buf, pos)
    if m:
        end = m.end()
        item = buf[pos:end].strip()
        output.write('"{}"'.format(item))
        if buf[end : end + 1] not in ("", "\r", "\n", "#"):
            raise Bareword(
                buf,
                pos,
                "The parser is trying its very best but could only make out '{}', but there is other junk on that line. You fix it.".format(
                    item
                ),
            )
        elif buf[end : end + 1] == "#":
            output.write(" ")

        return item, m.end()
    raise Bareword(
        buf,
        pos,
        "The parser doesn't know how to parse anymore and has given up. Use less barewords: {}...".format(
            repr(buf[pos : pos + 5])
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SafeYAML Linter, checks (or formats) a YAML file for common ambiguities"
    )

    parser.add_argument(
        "file",
        nargs="*",
        default=None,
        help="filename to read, without will read from stdin",
    )
    parser.add_argument(
        "--force-string-keys",
        action="store_true",
        default=False,
        help="quote every bareword",
    )
    parser.add_argument(
        "--force-commas", action="store_true", default=False, help="trailing commas"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="don't print cleaned file"
    )
    parser.add_argument(
        "--in-place", action="store_true", default=False, help="edit file"
    )

    parser.add_argument(
        "--json", action="store_true", default=False, help="output json instead of yaml"
    )

    args = parser.parse_args()  # will only return when action is given

    options = Options(
        force_string_keys=args.force_string_keys, force_commas=args.force_commas
    )

    if args.in_place:
        if args.json:
            print("error: safeyaml --in-place cannot be used with --json")
            print()
            sys.exit(-2)

        if len(args.file) < 1:
            print("error: safeyaml --in-place takes at least one file")
            print()
            sys.exit(-2)

        for filename in args.file:
            with open(filename, "r+") as fh:
                try:
                    output = io.StringIO()
                    obj = parse(fh.read(), output=output, options=options)
                except ParserErr as p:
                    line, col = get_position(p.buf, p.pos)
                    print(
                        "{}:{}:{}:{}".format(filename, line, col, p.explain()),
                        file=sys.stderr,
                    )
                    sys.exit(-2)
                else:
                    fh.seek(0)
                    fh.truncate(0)
                    fh.write(output.getvalue())

    else:
        input_fh, output_fh = sys.stdin, sys.stdout
        filename = "<stdin>"

        if args.file:
            if len(args.file) > 1:
                print(
                    "error: safeyaml only takes one file as argument, unless --in-place given"
                )
                print()
                sys.exit(-1)

            input_fh = open(args.file[0])  # closed on exit
            filename = args.file

        try:
            output = io.StringIO()
            obj = parse(input_fh.read(), output=output, options=options)
        except ParserErr as p:
            line, col = get_position(p.buf, p.pos)
            print(
                "{}:{}:{}:{}".format(filename, line, col, p.explain()), file=sys.stderr
            )
            sys.exit(-2)

        if not args.quiet:

            if args.json:
                json.dump(obj, output_fh)
            else:
                output_fh.write(output.getvalue())

    sys.exit(0)
