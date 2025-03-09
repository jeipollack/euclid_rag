#
# This file is part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fetch and process JIRA ticket data, extract relevant details, and
save results in a structured JSON format.
"""

import json
import time
from functools import reduce
from pathlib import Path
from typing import Any

import requests


def get_jira_issue(issue_name: str, email: str, api_token: str) -> tuple:
    """Get the JIRA issue data from the JIRA API."""
    url = f"https://rubinobs.atlassian.net/rest/api/latest/issue/{issue_name}"
    auth = requests.auth.HTTPBasicAuth(email, api_token)
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, auth=auth, headers=headers, timeout=10)

    if response.status_code == 200:
        return response.json(), None
    if response.status_code == 429:
        return [], f"{issue_name}: {response.status_code}"
    else:
        return None, f"{issue_name}: {response.status_code}"


def extract_reviewer_from_customfield(jira_data: dict) -> list:
    """Extract the reviewer(s) from the customfield_10048."""
    # Extract reviewer information from customfield_10048 if available
    reviewers = jira_data["fields"].get("customfield_10048", [])
    if reviewers:
        # Extract the display name(s) of the reviewer(s) into a list
        return [
            reviewer.get("displayName", "Reviewer not found")
            for reviewer in reviewers
        ]
    return ["No reviewer assigned"]


def extract_related_issues(jira_data: dict) -> list:
    """Extract the related issues from the JIRA data."""
    related_issues = []
    issue_links = jira_data["fields"].get("issuelinks", [])

    for link in issue_links:
        issue_relation = {}
        if "inwardIssue" in link:
            issue_relation = {
                "key": link["inwardIssue"].get("key", ""),
                "summary": link["inwardIssue"]["fields"].get(
                    "summary", "No summary"
                ),
                "status": link["inwardIssue"]["fields"]["status"].get(
                    "name", "Unknown status"
                ),
                "relationship": link["type"].get(
                    "outward", "Unknown relation"
                ),
            }
        elif "outwardIssue" in link:
            issue_relation = {
                "key": link["outwardIssue"].get("key", ""),
                "summary": link["outwardIssue"]["fields"].get(
                    "summary", "No summary"
                ),
                "status": link["outwardIssue"]["fields"]["status"].get(
                    "name", "Unknown status"
                ),
                "relationship": link["type"].get(
                    "outward", "Unknown relation"
                ),
            }
        related_issues.append(issue_relation)

    return related_issues


def extract_components(jira_data: dict) -> list:
    """Extract the components from the JIRA data."""
    components = jira_data["fields"].get("components", [])
    return [component.get("name", "No component") for component in components]


def extract_comments(jira_data: dict) -> list:
    """Extract the comments from the JIRA data."""
    comments = jira_data["fields"].get("comment", {}).get("comments", [])
    return [
        {
            "author": comment["author"].get("displayName", "Unknown author"),
            "body": comment.get("body", "No comment body"),
        }
        for comment in comments
    ]


def extract_parent_issue(jira_data: dict) -> dict:
    """Extract the parent issue from the JIRA data."""
    parent_issue = jira_data["fields"].get("parent", None)
    if parent_issue:
        return {
            "key": parent_issue.get("key", ""),
            "summary": parent_issue["fields"].get("summary", "No summary"),
            "status": parent_issue["fields"]["status"].get(
                "name", "Unknown status"
            ),
        }
    return {}


def safe_get(
    dictionary: dict[str, Any], keys: list, default: Any | None = None
) -> Any:
    """Safely extract information from nested dictionaries."""
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys,
        dictionary,
    )


def reformat_jira_data(jira_data: dict, ticket: str) -> dict:
    """Reformat the JIRA data into a simplified dictionary."""
    if not jira_data:
        # If jira_data is None or empty, return a default
        # dictionary with the error message
        return {
            "key": ticket,
            "summary": "No data available",
            "description": "Unauthorized or no data available",
            "status": "Unknown",
            "assignee": "Unassigned",
            "reviewers": ["No reviewer assigned"],
            "reporter": "Unknown",
            "created": "N/A",
            "updated": "N/A",
            "resolution": "Unresolved",
            "labels": [],
            "attachments": [],
            "comments": [{"author": "Unknown", "body": "No comments"}],
            "parent_issue": None,
            "related_issues": [],
            "components": ["No components"],
            "team": "No team",
            "project": "No project",
        }

    return {
        "key": jira_data.get("key", ""),
        "summary": jira_data["fields"].get("summary", ""),
        "description": jira_data["fields"].get("description", ""),
        "status": jira_data["fields"]["status"].get("name", "Unknown"),
        "assignee": safe_get(
            jira_data["fields"], ["assignee", "displayName"], "Unassigned"
        ),
        "reviewers": extract_reviewer_from_customfield(jira_data),
        "reporter": safe_get(
            jira_data["fields"], ["reporter", "displayName"], "Unknown"
        ),
        "created": jira_data["fields"].get("created", ""),
        "updated": jira_data["fields"].get("updated", ""),
        "resolution": safe_get(
            jira_data["fields"], ["resolution", "name"], "Unresolved"
        ),
        "labels": jira_data["fields"].get("labels", []),
        "attachments": [
            {
                "filename": attachment.get("filename", ""),
                "url": attachment.get("content", ""),
            }
            for attachment in jira_data["fields"].get("attachment", [])
        ],
        "comments": extract_comments(jira_data),
        "parent_issue": extract_parent_issue(
            jira_data
        ),  # Parent issue extraction
        "related_issues": extract_related_issues(
            jira_data
        ),  # Related issues extraction
        "components": extract_components(jira_data),  # Extract components
        "team": safe_get(
            jira_data["fields"], ["customfield_10056", "value"], "No team"
        ),
        "project": safe_get(
            jira_data["fields"], ["project", "name"], "No project"
        ),
    }


def write_to_file(
    results: dict,
    folder: str = "/Users/gmegias/Desktop/LSST_Developer/JIRA tickets",
) -> None:
    """Write the JIRA ticket data to a JSON file."""
    # Extract the prefix from the key (letters before '-')
    ticket_key = results["key"]  # Assuming 'key' is something like 'DM-12345'
    prefix = ticket_key.split("-")[0]  # Get the letters before the '-'

    # Construct the new folder path by appending the prefix
    folder_with_prefix = Path(folder / prefix)

    # Ensure the folder exists
    if not Path.exists(folder_with_prefix):
        Path.mkdir(folder_with_prefix)

    ticket_key = results["key"]  # Assuming 'key' is the Jira ticket key
    file_path = Path(
        folder_with_prefix / f"{ticket_key}.json"
    )  # Create the JSON file path

    # Write the individual result to a JSON file
    with Path.open(file_path, "w") as f:
        json.dump(
            results, f, indent=4
        )  # Writing with indentation for readability


def fetch_ticket(ticket: str, email: str, api_token: str) -> tuple:
    """Fetch the ticket data from JIRA."""
    jira_data, error_message = get_jira_issue(ticket, email, api_token)
    return reformat_jira_data(jira_data, ticket), error_message


def retry_fetch_ticket(
    ticket: str, email: str, api_token: str, max_retries: int = 5
) -> tuple:
    """Fetch the ticket with retry logic."""
    for attempt in range(max_retries):
        try:
            result, error_message = fetch_ticket(ticket, email, api_token)
        except Exception:
            if attempt + 1 == max_retries:
                raise  # Raise the error if max retries reached
            time.sleep(2**attempt + 2)  # Exponential backoff
        else:
            return result, error_message
    return None, "Failed to fetch ticket"
