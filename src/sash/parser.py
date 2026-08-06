"""Recursive-descent parser for the POSIX shell grammar (XCU 2.10).

One-token lookahead plus word inspection (two tokens only for the
`name ( )` function-definition form). Reserved words are promoted
positionally: a word counts as a reserved word only when it is a single
unquoted literal appearing where the grammar recognizes it, so `"if"` and
`\\if` are ordinary words.

Command substitutions captured by the lexer as raw text are recursively
parsed here (a staged re-read run eagerly, since the site and its contents
are statically known); here-document bodies, collected by the lexer at
newlines, get the same treatment after the main parse.
"""

from __future__ import annotations

from .lexer import Lexer
from .nodes import (
    AndOr,
    Arith,
    Assign,
    Async,
    BraceGroup,
    Case,
    CaseItem,
    CmdSub,
    Command,
    DQuote,
    For,
    FunDef,
    HereDoc,
    IdKind,
    If,
    Lit,
    Loc,
    Param,
    Pipeline,
    Program,
    Redirect,
    ShId,
    ShParseError,
    Simple,
    Subshell,
    TokEof,
    TokIoNumber,
    TokNewline,
    TokOp,
    TokWord,
    Token,
    Until,
    While,
    Word,
    WordPart,
    is_valid_name,
    word_single_unquoted_lit,
    word_to_reserved,
)

REDIR_OPS = frozenset(["<", ">", ">>", "<<", "<<-", "<&", ">&", "<>", ">|"])

_BAD_COMMAND_START = frozenset(
    ["then", "elif", "else", "fi", "do", "done", "esac", "in", "}"]
)


def parse(
    text: str,
    src: str = "<sh>",
    *,
    line: int = 1,
    col: int = 0,
    synthetic: bool = False,
) -> Program:
    return _Parser(text, src, line=line, col=col, synthetic=synthetic).parse_program()


class _Parser:
    def __init__(
        self,
        text: str,
        src: str,
        *,
        line: int = 1,
        col: int = 0,
        synthetic: bool = False,
    ) -> None:
        self.src = src
        self.lexer = Lexer(text, src, line=line, col=col, synthetic=synthetic)
        self.program_loc = Loc(src, line, col, 0, len(text), synthetic)
        self._peeked: Token | None = None
        self.tok: Token = self.lexer.next_token()
        self._heredocs: list[HereDoc] = []

    # -- token plumbing ------------------------------------------------------

    def _advance(self) -> None:
        if self._peeked is not None:
            self.tok = self._peeked
            self._peeked = None
        else:
            self.tok = self.lexer.next_token()

    def _peek2(self) -> Token:
        if self._peeked is None:
            self._peeked = self.lexer.next_token()
        return self._peeked

    def _tok_loc(self) -> Loc:
        return self.tok.loc

    def _error(self, message: str) -> ShParseError:
        return ShParseError(message, self._tok_loc())

    def _at_op(self, *syms: str) -> bool:
        return isinstance(self.tok, TokOp) and self.tok.id.sym in syms

    def _reserved(self) -> str | None:
        if isinstance(self.tok, TokWord):
            return word_to_reserved(self.tok.word)
        return None

    def _skip_newlines(self) -> None:
        while isinstance(self.tok, TokNewline):
            self._advance()

    def _take_word(self, what: str) -> Word:
        if not isinstance(self.tok, TokWord):
            raise self._error(f"expected {what}")
        word = self.tok.word
        self._advance()
        self._fill_parts(word.parts)
        return word

    def _expect_op(self, sym: str) -> ShId:
        if not self._at_op(sym):
            raise self._error(f"expected {sym!r}")
        assert isinstance(self.tok, TokOp)
        op_id = self.tok.id
        self._advance()
        return op_id

    def _expect_reserved(self, name: str) -> ShId:
        if self._reserved() != name:
            raise self._error(f"expected {name!r}")
        assert isinstance(self.tok, TokWord)
        kw = ShId(name, IdKind.OPERATOR, self.tok.word.loc)
        self._advance()
        return kw

    # -- nested command substitutions ---------------------------------------

    def _fill_parts(self, parts: list[WordPart] | None) -> None:
        if parts is None:
            return
        for part in parts:
            if isinstance(part, CmdSub):
                if part.program is None:
                    base = part.raw_loc if part.raw_loc is not None else part.loc
                    part.program = parse(
                        part.raw,
                        self.src,
                        line=base.line,
                        col=base.col,
                        synthetic=base.synthetic,
                    )
            elif isinstance(part, DQuote):
                self._fill_parts(list(part.parts))
            elif isinstance(part, Param):
                self._fill_parts(part.word)
            elif isinstance(part, Arith):
                for sub in part.parts:
                    if not isinstance(sub, ShId):
                        self._fill_parts([sub])

    # -- program structure ---------------------------------------------------

    def parse_program(self) -> Program:
        commands: list[Command] = []
        self._skip_newlines()
        while not isinstance(self.tok, TokEof):
            commands.extend(self._parse_complete_command())
            self._skip_newlines()
        for hd in self._heredocs:
            self._fill_parts(hd.body)
        return Program(commands, self.program_loc)

    def _parse_complete_command(self) -> list[Command]:
        result: list[Command] = []
        while True:
            cmd = self._parse_and_or()
            if self._at_op("&"):
                assert isinstance(self.tok, TokOp)
                cmd = Async(cmd, self.tok.id, cmd.loc)
                self._advance()
            elif self._at_op(";"):
                self._advance()
            else:
                result.append(cmd)
                break
            result.append(cmd)
            if isinstance(self.tok, (TokNewline, TokEof)):
                break
        if isinstance(self.tok, TokNewline):
            self._advance()
        elif not isinstance(self.tok, TokEof):
            raise self._error(f"unexpected token after command")
        return result

    def _parse_compound_list(self, stop: frozenset[str]) -> list[Command]:
        commands: list[Command] = []
        self._skip_newlines()
        while True:
            if self._at_stop(stop):
                return commands
            if isinstance(self.tok, TokEof):
                raise self._error(
                    f"unexpected end of input (expected one of {sorted(stop)})"
                )
            cmd = self._parse_and_or()
            if self._at_op("&"):
                assert isinstance(self.tok, TokOp)
                cmd = Async(cmd, self.tok.id, cmd.loc)
                self._advance()
            elif self._at_op(";"):
                self._advance()
            commands.append(cmd)
            self._skip_newlines()

    def _at_stop(self, stop: frozenset[str]) -> bool:
        reserved = self._reserved()
        if reserved is not None and reserved in stop:
            return True
        return isinstance(self.tok, TokOp) and self.tok.id.sym in stop

    # -- pipelines and lists -------------------------------------------------

    def _parse_and_or(self) -> Command:
        left = self._parse_pipeline()
        while self._at_op("&&", "||"):
            assert isinstance(self.tok, TokOp)
            op_id = self.tok.id
            self._advance()
            self._skip_newlines()
            right = self._parse_pipeline()
            left = AndOr(left, op_id, right, left.loc)
        return left

    def _parse_pipeline(self) -> Command:
        bang_id: ShId | None = None
        if self._reserved() == "!":
            assert isinstance(self.tok, TokWord)
            bang_id = ShId("!", IdKind.OPERATOR, self.tok.word.loc)
            self._advance()
        first = self._parse_command()
        cmds = [first]
        pipe_ids: list[ShId] = []
        while self._at_op("|"):
            assert isinstance(self.tok, TokOp)
            pipe_ids.append(self.tok.id)
            self._advance()
            self._skip_newlines()
            cmds.append(self._parse_command())
        if bang_id is None and not pipe_ids:
            return first
        return Pipeline(bang_id, cmds, pipe_ids, first.loc)

    # -- commands ------------------------------------------------------------

    def _parse_command(self) -> Command:
        if self._at_op("("):
            return self._parse_subshell()
        reserved = self._reserved()
        if reserved is not None:
            if reserved == "if":
                return self._parse_if()
            if reserved == "while":
                return self._parse_while_until(is_while=True)
            if reserved == "until":
                return self._parse_while_until(is_while=False)
            if reserved == "for":
                return self._parse_for()
            if reserved == "case":
                return self._parse_case()
            if reserved == "{":
                return self._parse_brace_group()
            if reserved in _BAD_COMMAND_START:
                raise self._error(f"unexpected reserved word {reserved!r}")
            # 'in', '!' handled elsewhere; fall through is impossible
        if isinstance(self.tok, TokWord):
            name = word_single_unquoted_lit(self.tok.word)
            if (
                name is not None
                and is_valid_name(name)
                and isinstance(self._peek2(), TokOp)
                and self._peek2().id.sym == "("  # type: ignore[union-attr]
            ):
                return self._parse_fundef(name)
        return self._parse_simple()

    def _parse_fundef(self, name: str) -> FunDef:
        assert isinstance(self.tok, TokWord)
        name_id = ShId(name, IdKind.FUNCTION_NAME, self.tok.word.loc)
        loc = self.tok.word.loc
        self._advance()
        lparen = self._expect_op("(")
        rparen = self._expect_op(")")
        self._skip_newlines()
        body = self._parse_command()
        if isinstance(body, Simple):
            raise ShParseError("function body must be a compound command", body.loc)
        return FunDef(name_id, body, [], [lparen, rparen], loc)

    def _parse_subshell(self) -> Subshell:
        lparen = self._expect_op("(")
        loc = lparen.loc
        body = self._parse_compound_list(frozenset({")"}))
        rparen = self._expect_op(")")
        redirects = self._parse_trailing_redirects()
        return Subshell(body, redirects, [lparen, rparen], loc)

    def _parse_brace_group(self) -> BraceGroup:
        lbrace = self._expect_reserved("{")
        body = self._parse_compound_list(frozenset({"}"}))
        rbrace = self._expect_reserved("}")
        redirects = self._parse_trailing_redirects()
        return BraceGroup(body, redirects, [lbrace, rbrace], lbrace.loc)

    def _parse_if(self) -> If:
        kw_if = self._expect_reserved("if")
        keywords = [kw_if]
        clauses: list[tuple[list[Command], list[Command]]] = []
        test = self._parse_compound_list(frozenset({"then"}))
        keywords.append(self._expect_reserved("then"))
        then_body = self._parse_compound_list(frozenset({"elif", "else", "fi"}))
        clauses.append((test, then_body))
        else_body: list[Command] | None = None
        while True:
            reserved = self._reserved()
            if reserved == "elif":
                keywords.append(self._expect_reserved("elif"))
                test = self._parse_compound_list(frozenset({"then"}))
                keywords.append(self._expect_reserved("then"))
                body = self._parse_compound_list(frozenset({"elif", "else", "fi"}))
                clauses.append((test, body))
                continue
            if reserved == "else":
                keywords.append(self._expect_reserved("else"))
                else_body = self._parse_compound_list(frozenset({"fi"}))
            keywords.append(self._expect_reserved("fi"))
            break
        redirects = self._parse_trailing_redirects()
        return If(clauses, else_body, redirects, keywords, kw_if.loc)

    def _parse_while_until(self, *, is_while: bool) -> Command:
        kw = self._expect_reserved("while" if is_while else "until")
        test = self._parse_compound_list(frozenset({"do"}))
        kw_do = self._expect_reserved("do")
        body = self._parse_compound_list(frozenset({"done"}))
        kw_done = self._expect_reserved("done")
        redirects = self._parse_trailing_redirects()
        keywords = [kw, kw_do, kw_done]
        if is_while:
            return While(test, body, redirects, keywords, kw.loc)
        return Until(test, body, redirects, keywords, kw.loc)

    def _parse_for(self) -> For:
        kw_for = self._expect_reserved("for")
        keywords = [kw_for]
        if not isinstance(self.tok, TokWord):
            raise self._error("expected a variable name after 'for'")
        name = word_single_unquoted_lit(self.tok.word)
        if name is None or not is_valid_name(name):
            raise self._error("expected a variable name after 'for'")
        var_id = ShId(name, IdKind.FOR_VARIABLE, self.tok.word.loc)
        self._advance()
        self._skip_newlines()
        in_words: list[Word] | None = None
        if self._reserved() == "in":
            keywords.append(self._expect_reserved("in"))
            in_words = []
            while isinstance(self.tok, TokWord):
                in_words.append(self._take_word("word"))
            if self._at_op(";"):
                self._advance()
            elif isinstance(self.tok, TokNewline):
                pass
            else:
                raise self._error("expected ';' or newline after 'for' word list")
        elif self._at_op(";"):
            self._advance()
        self._skip_newlines()
        keywords.append(self._expect_reserved("do"))
        body = self._parse_compound_list(frozenset({"done"}))
        keywords.append(self._expect_reserved("done"))
        redirects = self._parse_trailing_redirects()
        return For(var_id, in_words, body, redirects, keywords, kw_for.loc)

    def _parse_case(self) -> Case:
        kw_case = self._expect_reserved("case")
        keywords = [kw_case]
        subject = self._take_word("word after 'case'")
        self._skip_newlines()
        keywords.append(self._expect_reserved("in"))
        self._skip_newlines()
        items: list[CaseItem] = []
        while self._reserved() != "esac":
            if isinstance(self.tok, TokEof):
                raise self._error("unexpected end of input (expected 'esac')")
            items.append(self._parse_case_item())
            self._skip_newlines()
        keywords.append(self._expect_reserved("esac"))
        redirects = self._parse_trailing_redirects()
        return Case(subject, items, redirects, keywords, kw_case.loc)

    def _parse_case_item(self) -> CaseItem:
        keywords: list[ShId] = []
        loc = self._tok_loc()
        if self._at_op("("):
            keywords.append(self._expect_op("("))
        patterns = [self._take_word("case pattern")]
        while self._at_op("|"):
            self._advance()
            patterns.append(self._take_word("case pattern"))
        keywords.append(self._expect_op(")"))
        body = self._parse_compound_list(frozenset({";;", "esac"}))
        if self._at_op(";;"):
            keywords.append(self._expect_op(";;"))
            self._skip_newlines()
        return CaseItem(patterns, body, keywords, loc)

    # -- simple commands and redirections ------------------------------------

    def _parse_simple(self) -> Simple:
        loc = self._tok_loc()
        assigns: list[Assign] = []
        words: list[Word] = []
        redirects: list[Redirect] = []
        cmd_id: ShId | None = None
        prefix = True
        while True:
            tok = self.tok
            if isinstance(tok, TokIoNumber):
                n = tok.n
                self._advance()
                redirects.append(self._parse_redirect(n))
                continue
            if isinstance(tok, TokOp) and tok.id.sym in REDIR_OPS:
                redirects.append(self._parse_redirect(None))
                continue
            if isinstance(tok, TokWord):
                if prefix:
                    assign = self._try_assignment(tok.word)
                    if assign is not None:
                        assigns.append(assign)
                        self._advance()
                        continue
                    prefix = False
                    text = word_single_unquoted_lit(tok.word)
                    if text is not None:
                        cmd_id = ShId(text, IdKind.COMMAND, tok.word.loc)
                words.append(self._take_word("word"))
                continue
            break
        if not assigns and not words and not redirects:
            raise self._error("expected a command")
        return Simple(assigns, words, cmd_id, redirects, loc)

    def _try_assignment(self, word: Word) -> Assign | None:
        if not word.parts or not isinstance(word.parts[0], Lit):
            return None
        text = word.parts[0].text
        eq = text.find("=")
        if eq <= 0:
            return None
        name = text[:eq]
        if not is_valid_name(name):
            return None
        first_loc = word.parts[0].loc
        name_loc = Loc(
            first_loc.src,
            first_loc.line,
            first_loc.col,
            first_loc.pos,
            eq,
            first_loc.synthetic,
        )
        value_parts: list[WordPart] = list(word.parts[1:])
        rest = text[eq + 1 :]
        if rest:
            rest_loc = Loc(
                first_loc.src,
                first_loc.line,
                first_loc.col + eq + 1,
                first_loc.pos + eq + 1,
                len(rest),
                first_loc.synthetic,
            )
            value_parts.insert(0, Lit(rest, rest_loc))
        self._fill_parts(value_parts)
        name_id = ShId(name, IdKind.VARIABLE_BINDER, name_loc)
        return Assign(name_id, Word(value_parts, word.loc), word.loc)

    def _parse_redirect(self, n: int | None) -> Redirect:
        assert isinstance(self.tok, TokOp)
        op_id = self.tok.id
        self._advance()
        word = self._take_word(f"target of {op_id.sym!r}")
        target: Word | HereDoc = word
        if op_id.sym in ("<<", "<<-"):
            hd = self.lexer.heredoc_for(word)
            if hd is None:
                raise ShParseError("missing here-document", op_id.loc)
            self._heredocs.append(hd)
            target = hd
        return Redirect(n, op_id, target, op_id.loc)

    def _parse_trailing_redirects(self) -> list[Redirect]:
        redirects: list[Redirect] = []
        while True:
            if isinstance(self.tok, TokIoNumber):
                n = self.tok.n
                self._advance()
                redirects.append(self._parse_redirect(n))
                continue
            if isinstance(self.tok, TokOp) and self.tok.id.sym in REDIR_OPS:
                redirects.append(self._parse_redirect(None))
                continue
            return redirects
