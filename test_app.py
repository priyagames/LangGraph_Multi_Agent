import unittest
from unittest.mock import patch

from app import _format_result, app
from jobs_db import JobsDatabaseError


class AppDatabaseErrorHandlingTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("app.graph.invoke", side_effect=JobsDatabaseError("The job database is unavailable right now. Please try again in a moment."))
    def test_chat_returns_friendly_error_when_db_is_unavailable(self, _mock_invoke):
        response = self.client.post("/chat", json={"message": "get me 2 recent jobs"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["done"])
        self.assertIn("database", payload["reply"].lower())
        self.assertIn("unavailable", payload["reply"].lower())

    def test_exp_validation_accepts_months_as_plausible(self):
        from jd_intake_pipeline import _validate_single_field

        result = _validate_single_field("exp", "2 Months")
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "")

    def test_format_result_uses_clear_sections_for_jobs(self):
        formatted = _format_result({
            "status": "answered",
            "query_result": [{
                "_id": "68b7d1d4c0f94f0012345678",
                "job_id": "acme_2043",
                "title": "Senior Backend Engineer",
                "company": "Acme",
                "location": "Bangalore",
                "exp": "5+ years",
                "work_mode": "Hybrid",
            }],
        })
        self.assertIn("Recent jobs", formatted)
        self.assertIn("Senior Backend Engineer", formatted)
        self.assertIn("Job ID: acme_2043", formatted)
        self.assertIn("Company: Acme", formatted)

    def test_publish_job_in_db_generates_human_friendly_job_id(self):
        from jobs_db import _generate_job_id, _slug_job_id

        self.assertEqual(_slug_job_id("Acme Corp"), "ACME-CORP")
        generated = _generate_job_id({}, "Acme Corp")
        self.assertTrue(generated.startswith("ACME-CORP-"))
        self.assertRegex(generated, r"^ACME-CORP-\d+$")

    def test_format_result_mentions_resume_processing_success(self):
        formatted = _format_result({
            "status": "qualified",
            "target_job": {"title": "Senior Backend Engineer"},
            "qualify_score": 88,
            "qualify_feedback": "Strong backend fit with good API and DB experience.",
        })
        self.assertIn("Resume uploaded and processed successfully", formatted)
        self.assertIn("Score: 88/100", formatted)

    def test_chat_gives_direct_delete_and_thanks_response(self):
        response = self.client.post("/chat", json={"message": "delete jobs"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("job id", payload["reply"].lower())

        response = self.client.post("/chat", json={"message": "thank you"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("you're welcome", payload["reply"].lower())

    def test_delete_and_update_accept_only_exact_job_id(self):
        from jobs_db import delete_job_by_id, update_job_by_id

        with self.assertRaises(ValueError):
            delete_job_by_id("")

        with self.assertRaises(ValueError):
            update_job_by_id("", {"title": "New title"})


if __name__ == "__main__":
    unittest.main()
