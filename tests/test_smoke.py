import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "client"))

import main as server_main  # noqa: E402
from url_utils import normalize_url_for_open  # noqa: E402


class CommandRoutingSmokeTests(unittest.TestCase):
    def test_open_google_becomes_url_action(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("open the google"),
            {"command": "open_url", "url": "https://www.google.com"},
        )

    def test_open_goodgle_becomes_google_url_action(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("open the goodgle"),
            {"command": "open_url", "url": "https://www.google.com"},
        )

    def test_open_google_in_new_tab_becomes_url_action(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("open google in a new tab"),
            {"command": "open_url", "url": "https://www.google.com"},
        )

    def test_open_google_search_becomes_url_action(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("open google search"),
            {"command": "open_url", "url": "https://www.google.com"},
        )

    def test_open_app_becomes_direct_app_action(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("open chrome"),
            {"command": "open_app", "app": "Google Chrome"},
        )

    def test_open_safari_and_login_to_leetcode_starts_in_browser(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions(
                "open safari and login to my leetcode"
            ),
            [
                {"command": "open_app", "app": "Safari"},
                {
                    "command": "open_url",
                    "url": "https://leetcode.com/accounts/login/",
                    "browser": "Safari",
                },
            ],
        )

    def test_deepgram_lead_code_transcript_starts_leetcode_login(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions(
                "Open Safari and log in to my lead code."
            ),
            [
                {"command": "open_app", "app": "Safari"},
                {
                    "command": "open_url",
                    "url": "https://leetcode.com/accounts/login/",
                    "browser": "Safari",
                },
            ],
        )

    def test_open_safari_and_leetcode_starts_in_browser(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions("open safari and open leetcode"),
            [
                {"command": "open_app", "app": "Safari"},
                {
                    "command": "open_url",
                    "url": "https://leetcode.com",
                    "browser": "Safari",
                },
            ],
        )

    def test_multiple_open_sites_start_in_spoken_order(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions(
                "open youtube and then open google"
            ),
            [
                {"command": "open_url", "url": "https://www.youtube.com"},
                {"command": "open_url", "url": "https://www.google.com"},
            ],
        )

    def test_three_open_sites_start_in_spoken_order(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions(
                "open github then open youtube and open google"
            ),
            [
                {"command": "open_url", "url": "https://github.com"},
                {"command": "open_url", "url": "https://www.youtube.com"},
                {"command": "open_url", "url": "https://www.google.com"},
            ],
        )

    def test_multi_step_leetcode_command_starts_all_requested_sites(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions(
                "open google and open leetcode and solve today's daily problem"
            ),
            [
                {"command": "open_url", "url": "https://www.google.com"},
                {"command": "open_url", "url": "https://leetcode.com"},
            ],
        )

    def test_longer_site_alias_wins_over_shorter_alias(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions("open google docs"),
            [{"command": "open_url", "url": "https://docs.google.com"}],
        )

    def test_open_safari_and_navigate_to_leetcode_problem_starts_direct_problem_url(self) -> None:
        self.assertEqual(
            server_main.deterministic_start_actions(
                "open leetcode in safari and navigate to problem 1. Two Sum"
            ),
            [
                {"command": "open_app", "app": "Safari"},
                {
                    "command": "open_url",
                    "url": "https://leetcode.com/problems/two-sum/",
                    "browser": "Safari",
                },
            ],
        )

    def test_open_new_tab_becomes_hotkey(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("open new tab"),
            {"command": "press_hotkey", "keys": ["command", "t"]},
        )

    def test_close_tab_becomes_hotkey(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("close current tab"),
            {"command": "press_hotkey", "keys": ["command", "w"]},
        )

    def test_switch_tab_becomes_hotkey(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("go to the next tab"),
            {"command": "press_hotkey", "keys": ["ctrl", "tab"]},
        )

    def test_reopen_closed_tab_becomes_hotkey(self) -> None:
        self.assertEqual(
            server_main.deterministic_simple_action("reopen closed tab"),
            {"command": "press_hotkey", "keys": ["command", "shift", "t"]},
        )

    def test_multi_step_command_stays_agentic(self) -> None:
        self.assertIsNone(
            server_main.deterministic_simple_action(
                "open google and open leetcode and solve today's daily problem"
            )
        )

    def test_leetcode_task_detection_and_guidance(self) -> None:
        task = "open google and open leetcode and solve today's daily problem"
        self.assertTrue(server_main.is_leetcode_task(task))
        guidance = server_main.task_specific_guidance(task)
        self.assertIn("Accepted", guidance)
        self.assertIn("submit again", guidance)

    def test_leetcode_misspelling_still_triggers_agentic_mode(self) -> None:
        self.assertTrue(
            server_main.is_leetcode_task("solve the leedcofe daily problem")
        )

    def test_leetcode_request_is_not_misread_as_date_query(self) -> None:
        self.assertIsNone(
            server_main.local_simple_response(
                "solve today's daily problem on leetcode"
            )
        )

    def test_hey_bro_stop_is_shutdown_command(self) -> None:
        self.assertEqual(server_main.strip_wake_phrase("hey bro stop"), "stop")
        self.assertTrue(server_main.is_stop_command("stop"))

    def test_regular_command_is_not_shutdown_command(self) -> None:
        self.assertFalse(server_main.is_stop_command("open chrome"))


class UrlNormalizationSmokeTests(unittest.TestCase):
    def test_normalize_bare_site_name(self) -> None:
        self.assertEqual(
            normalize_url_for_open("google"),
            "https://www.google.com",
        )

    def test_normalize_site_with_tab_phrase(self) -> None:
        self.assertEqual(
            normalize_url_for_open("google in a new tab"),
            "https://www.google.com",
        )

    def test_normalize_spoken_domain(self) -> None:
        self.assertEqual(
            normalize_url_for_open("github dot com"),
            "https://github.com",
        )

    def test_normalize_common_google_misspelling(self) -> None:
        self.assertEqual(
            normalize_url_for_open("gioogle"),
            "https://www.google.com",
        )

    def test_normalize_leetcode_misspelling(self) -> None:
        self.assertEqual(
            normalize_url_for_open("leedcofe"),
            "https://leetcode.com",
        )

    def test_fallback_to_search_for_phrase(self) -> None:
        self.assertEqual(
            normalize_url_for_open("weather in delhi"),
            "https://www.google.com/search?q=weather+in+delhi",
        )


class ServerHealthSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_route_reports_ok_and_websocket_path(self) -> None:
        payload = await server_main.health()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["websocket"], "/ws/agent")


if __name__ == "__main__":
    unittest.main()
