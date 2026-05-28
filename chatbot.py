"""
=============================================================
MODULE 7: AI Chatbot — chatbot.py
AI-Powered Intelligent CMDB Management System
=============================================================
Natural-language CMDB query assistant powered by OpenAI.
Translates user questions into ServiceNow API queries,
fetches live CMDB data, and returns natural-language answers.
=============================================================
"""

import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Import CMDB query helper
# -----------------------------------------------------------
from insert_servicenow import query_cmdb

# -----------------------------------------------------------
# OpenAI Configuration
# -----------------------------------------------------------

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_MODEL = "gpt-4o-mini"

# -----------------------------------------------------------
# Supported intent → CMDB query mapping
# -----------------------------------------------------------

INTENT_PATTERNS = [

    {
        "keywords": ["linux server", "linux servers"],
        "table": "cmdb_ci_server",
        "query": "osLIKELinux^ORosLIKERed Hat^ORosLIKEUbuntu^ORosLIKECentOS",
        "fields": "name,ip_address,os,environment,manufacturer"
    },

    {
        "keywords": ["windows server", "windows servers"],
        "table": "cmdb_ci_server",
        "query": "osLIKEWindows",
        "fields": "name,ip_address,os,environment,manufacturer"
    },

    {
        "keywords": ["production server", "production servers", "prod server"],
        "table": "cmdb_ci_server",
        "query": "environment=production",
        "fields": "name,ip_address,os,environment,manufacturer"
    },

    {
        "keywords": ["production database", "prod database", "production db"],
        "table": "cmdb_ci_database",
        "query": "environment=production",
        "fields": "name,ip_address,os,environment"
    },

    {
        "keywords": ["all database", "databases", "all db"],
        "table": "cmdb_ci_database",
        "query": "",
        "fields": "name,ip_address,os,environment"
    },

    {
        "keywords": ["non-operational", "not operational", "offline"],
        "table": "cmdb_ci",
        "query": "operational_status!=1",
        "fields": "name,sys_class_name,operational_status,environment"
    },

    {
        "keywords": ["cloud resource", "cloud resources", "cloud vm"],
        "table": "cmdb_ci_vm_instance",
        "query": "",
        "fields": "name,ip_address,manufacturer,environment"
    },

    {
        "keywords": ["network device", "network devices", "firewall", "switch", "router"],
        "table": "cmdb_ci_netgear",
        "query": "",
        "fields": "name,ip_address,manufacturer,environment"
    },

    {
        "keywords": ["count server", "how many server", "server count"],
        "table": "cmdb_ci_server",
        "query": "",
        "fields": "name,environment",
        "aggregate": True
    },

    {
        "keywords": ["storage", "storage device"],
        "table": "cmdb_ci_storage_device",
        "query": "",
        "fields": "name,ip_address,manufacturer,environment"
    },

    {
        "keywords": ["application", "app", "all application"],
        "table": "cmdb_ci_appl",
        "query": "",
        "fields": "name,manufacturer,environment"
    },
]

# -----------------------------------------------------------
# Intent Detection
# -----------------------------------------------------------

def detect_intent(user_message: str):

    msg_lower = user_message.lower()

    for pattern in INTENT_PATTERNS:

        if any(kw in msg_lower for kw in pattern["keywords"]):
            return pattern

    return None

# -----------------------------------------------------------
# Format CMDB Records
# -----------------------------------------------------------

def _format_records(records: list, limit: int = 20):

    if not records:
        return "No records found."

    sample = records[:limit]

    lines = [json.dumps(r, default=str) for r in sample]

    result = "\n".join(lines)

    if len(records) > limit:
        result += f"\n... and {len(records) - limit} more records."

    return result

# -----------------------------------------------------------
# OpenAI API Call
# -----------------------------------------------------------

def _call_openai(system_prompt: str, user_prompt: str) -> str:

    try:

        response = client.chat.completions.create(

            model=OPENAI_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:

        logger.error(f"OpenAI API error: {e}")

        return f"I encountered an error communicating with OpenAI: {str(e)}"

# -----------------------------------------------------------
# Main Chatbot Handler
# -----------------------------------------------------------

def chat(user_message: str) -> str:

    """
    Process user message,
    query CMDB if needed,
    and return intelligent response.
    """

    logger.info(f"Chatbot received: {user_message}")

    intent = detect_intent(user_message)

    # -------------------------------------------------------
    # CMDB Query Based Questions
    # -------------------------------------------------------

    if intent:

        records = query_cmdb(

            table=intent["table"],

            sysparm_query=intent.get("query", ""),

            sysparm_fields=intent.get(
                "fields",
                "name,environment"
            ),

            sysparm_limit=100
        )

        # Aggregate Queries

        if intent.get("aggregate"):

            env_counts = {}

            for r in records:

                env = r.get("environment", "unknown")

                env_counts[env] = env_counts.get(env, 0) + 1

            data_context = (

                f"Total server count: {len(records)}\n"

                f"By environment: {json.dumps(env_counts)}"
            )

        else:

            data_context = _format_records(records)

        system_prompt = (

            "You are an expert ServiceNow CMDB assistant. "

            "You have retrieved live CMDB data from ServiceNow. "

            "Answer professionally and clearly. "

            "Summarize records when large. "

            "Provide useful infrastructure insights."
        )

        user_prompt = (

            f"User Question:\n{user_message}\n\n"

            f"CMDB Data Retrieved From Table "
            f"{intent['table']}:\n\n"

            f"{data_context}\n\n"

            "Answer the user professionally."
        )

    # -------------------------------------------------------
    # General Knowledge Questions
    # -------------------------------------------------------

    else:

        system_prompt = (

            "You are an expert ServiceNow CMDB and CSDM architect. "

            "Answer questions about CMDB, ServiceNow, CSDM, "

            "CI classes, infrastructure management, "

            "and enterprise best practices."
        )

        user_prompt = user_message

    return _call_openai(system_prompt, user_prompt)