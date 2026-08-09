#!/usr/bin/env python3
"""Unit tests for training/form_tag.py.

All examples here are synthetic (authored for this test file), never
corpus rows. Run with: python3 -m unittest discover training/tests
(or: python3 -m unittest training.tests.test_form_tag)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import form_tag as ft  # noqa: E402


class TestShellOnly(unittest.TestCase):
    """Plain shell statements/pipelines/chains — no accompanying prose."""

    def test_bare_invocation(self):
        self.assertEqual(ft.classify_form("kubectl get pods -n prod"), "shell_only")

    def test_pipeline(self):
        self.assertEqual(
            ft.classify_form("kubectl get pods -n prod | grep CrashLoop"),
            "shell_only",
        )

    def test_logical_chain(self):
        self.assertEqual(
            ft.classify_form(
                "kubectl scale deploy web --replicas=0 && kubectl get deploy web"
            ),
            "shell_only",
        )

    def test_multiline_script(self):
        text = (
            "kubectl get pvc -n data --no-headers | awk '{print $1}' | "
            "while read v; do kubectl delete pvc \"$v\" -n data; done"
        )
        self.assertEqual(ft.classify_form(text), "shell_only")

    def test_heredoc_manifest(self):
        text = (
            "kubectl apply -f - <<EOF\n"
            "apiVersion: v1\n"
            "kind: Namespace\n"
            "metadata:\n"
            "  name: preview\n"
            "EOF"
        )
        self.assertEqual(ft.classify_form(text), "shell_only")

    def test_command_substitution(self):
        self.assertEqual(
            ft.classify_form("kubectl delete pod $(kubectl get pod -o name)"),
            "shell_only",
        )


class TestQuotedPayloadIsShellish(unittest.TestCase):
    """Boundary rule: quoted payloads (even destructive-sounding words in
    data position) are SHELLISH, not NL — the wrapping command decides."""

    def test_echo_quoted_destructive_string(self):
        self.assertEqual(
            ft.classify_form("echo 'rm target.txt'"), "shell_only"
        )

    def test_read_prompt_with_command_text_inside(self):
        text = 'read -p "Scale legacy deployment to zero? kubectl scale deployment old-api --replicas=0" response'
        self.assertEqual(ft.classify_form(text), "shell_only")

    def test_json_patch_payload(self):
        text = "kubectl patch pvc data -p '{\"spec\":{\"resources\":{\"requests\":{\"storage\":\"5Gi\"}}}}'"
        self.assertEqual(ft.classify_form(text), "shell_only")

    def test_heredoc_body_is_payload_not_prose(self):
        text = (
            "cat <<EOF > runbook.txt\n"
            "In an emergency: kubectl delete deployment web -n prod\n"
            "EOF"
        )
        self.assertEqual(ft.classify_form(text), "shell_only")


class TestNestedInterpretersAreShellish(unittest.TestCase):
    """Boundary rule: nested interpreters are SHELLISH regardless of what
    the nested string contains."""

    def test_kubectl_exec_sh_c(self):
        text = 'kubectl exec pod-1 -- sh -c "rm -rf /data/tmp"'
        self.assertEqual(ft.classify_form(text), "shell_only")

    def test_psql_copy_to_program(self):
        text = "psql -c \"COPY orders TO PROGRAM 'rm -rf /'\""
        self.assertEqual(ft.classify_form(text), "shell_only")

    def test_bash_c(self):
        text = 'bash -c "kubectl delete ns staging"'
        self.assertEqual(ft.classify_form(text), "shell_only")


class TestShellProseComments(unittest.TestCase):
    """shell_prose: shell command(s) embedded in or accompanied by
    prose — ticket text, chat, or comments around a command."""

    def test_comment_only_line_wrapping_a_command(self):
        text = "# DO NOT run in prod: kubectl delete namespace payment-system"
        self.assertEqual(ft.classify_form(text), "shell_prose")

    def test_trailing_narrative_comment(self):
        text = 'kubectl edit deployment web -n prod  # set terminationGracePeriodSeconds: 0 and save'
        self.assertEqual(ft.classify_form(text), "shell_prose")

    def test_comment_line_plus_real_command(self):
        text = (
            "# old runbook step, no longer used: kubectl delete deployment web -n prod\n"
            "kubectl rollout status deployment/web -n prod"
        )
        self.assertEqual(ft.classify_form(text), "shell_prose")

    def test_quoted_hash_is_not_a_real_comment(self):
        # '#' inside a double-quoted echo payload is data, not a real
        # shell comment — stays shell_only.
        text = 'echo "# cleanup: kubectl delete pvc --all -n staging" >> notes.md'
        self.assertEqual(ft.classify_form(text), "shell_only")


class TestNlWhole(unittest.TestCase):
    """nl_whole: the ENTIRE statement is natural language, no shell
    syntax at all."""

    def test_given_example_question(self):
        self.assertEqual(
            ft.classify_form("can you bounce the payments deployment in staging?"),
            "nl_whole",
        )

    def test_bare_tool_mention_without_syntax(self):
        # Boundary rule: naming a tool alone isn't command syntax.
        self.assertEqual(
            ft.classify_form("restart the api pods with kubectl"),
            "nl_whole",
        )

    def test_question(self):
        self.assertEqual(
            ft.classify_form("is it safe to restart the database?"),
            "nl_whole",
        )

    def test_past_tense_report(self):
        self.assertEqual(
            ft.classify_form(
                "The migration from PVC-based storage to object storage "
                "was completed ahead of schedule"
            ),
            "nl_whole",
        )

    def test_conditional(self):
        self.assertEqual(
            ft.classify_form(
                "if we were to remove the networkpolicy in staging, "
                "would the egress tests still pass?"
            ),
            "nl_whole",
        )

    def test_domain_noun_is_not_a_command_mention(self):
        # "flux" the noun (as in FluxCD reconciliation), not an
        # invocation — no clause-boundary command token present.
        self.assertEqual(
            ft.classify_form(
                "can someone explain what the flux reconciliation did "
                "to the image tags on cluster west?"
            ),
            "nl_whole",
        )


class TestDirectiveFold(unittest.TestCase):
    """Design choice beyond the given rules: terse bare-imperative
    English commands (no literal syntax, no stated subject) read as a
    command spelled in words and fold into shell_prose, distinct from
    genuine narrative/question/report NL."""

    def test_bare_imperative_no_subject(self):
        self.assertEqual(ft.classify_form("restart the workers"), "shell_prose")

    def test_bare_imperative_delete(self):
        self.assertEqual(ft.classify_form("delete the api credentials"), "shell_prose")

    def test_please_imperative(self):
        self.assertEqual(
            ft.classify_form("Please rotate the payments certificate during the freeze."),
            "shell_prose",
        )

    def test_report_clause_plus_directive_stays_nl(self):
        # The classifier only reads the mood of the WHOLE statement (its
        # leading word), not each clause independently — a situational
        # report clause in front of an imperative tail keeps the overall
        # shape of a narrative sentence, so this stays nl_whole. Compare
        # to test_bare_imperative_no_subject, which has no such lead-in.
        text = (
            "The ingestion worker is failing again; restart it and retain "
            "all diagnostic records while investigating."
        )
        self.assertEqual(ft.classify_form(text), "nl_whole")

    def test_directive_beyond_word_cap_stays_nl(self):
        # Past ~9 words a directive typically carries a justification
        # clause and reads as a formal ticket description rather than
        # terse chatops shorthand.
        text = (
            "Decommission the legacy PostgreSQL statefulset after "
            "confirming the migration to the managed database completed"
        )
        self.assertEqual(ft.classify_form(text), "nl_whole")

    def test_subject_first_sentence_stays_nl(self):
        # Has an explicit subject ("someone") before the verb -> not a
        # bare imperative -> nl_whole.
        self.assertEqual(
            ft.classify_form("someone check if the metrics are working"),
            "nl_whole",
        )

    def test_gerund_lead_stays_nl(self):
        # "rolling back..." describes an ongoing action, not a command.
        self.assertEqual(
            ft.classify_form("rolling back postgres to the backup"),
            "nl_whole",
        )

    def test_past_tense_lead_stays_nl(self):
        self.assertEqual(
            ft.classify_form(
                "reverted the change that accidentally deleted the staging "
                "PVCs -- postmortem notes attached"
            ),
            "nl_whole",
        )


class TestInputValidation(unittest.TestCase):
    def test_none_raises(self):
        with self.assertRaises(ValueError):
            ft.classify_form(None)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ft.classify_form("   ")


if __name__ == "__main__":
    unittest.main()
