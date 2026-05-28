"""
=============================================================
MODULE 4: ServiceNow Integration — insert_servicenow.py
AI-Powered Intelligent CMDB Management System
=============================================================
Enterprise-grade REST API integration for:
- Create / Update (UPSERT)
- CMDB reconciliation
- Bulk onboarding
- Chatbot querying
=============================================================
"""

import os
import time
import logging
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# ServiceNow Configuration
# -----------------------------------------------------------

SN_INSTANCE = os.getenv(
    "SN_INSTANCE",
    "https://dev305829.service-now.com"
)

SN_USERNAME = os.getenv(
    "SN_USERNAME",
    "admin"
)

SN_PASSWORD = os.getenv("SN_PASSWORD")

# -----------------------------------------------------------
# API Defaults
# -----------------------------------------------------------

DEFAULT_TIMEOUT = 30

MAX_RETRIES = 3

BACKOFF_FACTOR = 0.5

RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# -----------------------------------------------------------
# CMDB Field Mapping
# -----------------------------------------------------------

FIELD_MAP = {

    "cmdb_ci_server": [

        "name",
        "ip_address",
        "os",
        "manufacturer",
        "description",
        "environment",
        "serial_number",
        "ram",
        "cpu_count"
    ],

    "cmdb_ci_database": [

        "name",
        "ip_address",
        "os",
        "manufacturer",
        "description",
        "environment",
        "version",
        "database_type"
    ],

    "cmdb_ci_netgear": [

        "name",
        "ip_address",
        "manufacturer",
        "description",
        "environment",
        "firmware_version"
    ],

    "cmdb_ci_vm_instance": [

        "name",
        "ip_address",
        "os",
        "manufacturer",
        "description",
        "environment",
        "cloud_provider",
        "region"
    ],

    "cmdb_ci_appl": [

        "name",
        "manufacturer",
        "description",
        "environment",
        "version",
        "install_directory"
    ],

    "cmdb_ci_storage_device": [

        "name",
        "ip_address",
        "manufacturer",
        "description",
        "environment",
        "disk_space"
    ],
}

# -----------------------------------------------------------
# Build HTTP Session
# -----------------------------------------------------------

def _build_session(username, password):

    session = requests.Session()

    session.auth = (username, password)

    session.headers = {

        "Content-Type": "application/json",

        "Accept": "application/json"
    }

    retry = Retry(

        total=MAX_RETRIES,

        backoff_factor=BACKOFF_FACTOR,

        status_forcelist=RETRY_STATUS_CODES,

        allowed_methods=["GET", "POST", "PATCH"]
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)

    session.mount("http://", adapter)

    return session

# -----------------------------------------------------------
# Build Payload
# -----------------------------------------------------------

def _build_payload(record, table):

    allowed = FIELD_MAP.get(table, list(record.keys()))

    payload = {

        k: v

        for k, v in record.items()

        if k in allowed and v not in [None, ""]
    }

    return payload

# -----------------------------------------------------------
# Query Existing CI
# -----------------------------------------------------------

def find_existing_ci(

    table,
    ci_name,
    instance=None,
    username=None,
    password=None
):

    inst = instance or SN_INSTANCE

    usr = username or SN_USERNAME

    pwd = password or SN_PASSWORD

    session = _build_session(usr, pwd)

    url = f"{inst}/api/now/table/{table}"

    params = {

        "sysparm_query": f"name={ci_name}",

        "sysparm_limit": 1
    }

    try:

        response = session.get(

            url,

            params=params,

            timeout=DEFAULT_TIMEOUT
        )

        response.raise_for_status()

        result = response.json().get("result", [])

        if result:

            return result[0]

        return None

    except Exception as e:

        logger.error(f"Error checking existing CI: {e}")

        return None

# -----------------------------------------------------------
# UPSERT CI
# -----------------------------------------------------------

def upsert_ci(

    record,
    table,
    instance=None,
    username=None,
    password=None
):

    inst = instance or SN_INSTANCE

    usr = username or SN_USERNAME

    pwd = password or SN_PASSWORD

    session = _build_session(usr, pwd)

    payload = _build_payload(record, table)

    ci_name = payload.get("name")

    if not ci_name:

        return {

            "status": "error",

            "message": "CI name missing",

            "table": table
        }

    # -------------------------------------------------------
    # STEP 1 — CHECK EXISTING CI
    # -------------------------------------------------------

    existing = find_existing_ci(

        table,

        ci_name,

        inst,

        usr,

        pwd
    )

    # -------------------------------------------------------
    # STEP 2 — UPDATE EXISTING
    # -------------------------------------------------------

    if existing:

        sys_id = existing["sys_id"]

        update_url = (

            f"{inst}/api/now/table/"
            f"{table}/{sys_id}"
        )

        try:

            logger.info(f"Updating existing CI: {ci_name}")

            response = session.patch(

                update_url,

                json=payload,

                timeout=DEFAULT_TIMEOUT
            )

            response.raise_for_status()

            return {

                "status": "success",

                "action": "UPDATED",

                "name": ci_name,

                "table": table,

                "sys_id": sys_id,

                "response_code": response.status_code
            }

        except Exception as e:

            logger.error(f"Update failed: {e}")

            return {

                "status": "error",

                "action": "UPDATE_FAILED",

                "name": ci_name,

                "table": table,

                "error": str(e)
            }

    # -------------------------------------------------------
    # STEP 3 — CREATE NEW CI
    # -------------------------------------------------------

    else:

        create_url = f"{inst}/api/now/table/{table}"

        try:

            logger.info(f"Creating new CI: {ci_name}")

            response = session.post(

                create_url,

                json=payload,

                timeout=DEFAULT_TIMEOUT
            )

            response.raise_for_status()

            result = response.json().get("result", {})

            return {

                "status": "success",

                "action": "CREATED",

                "name": ci_name,

                "table": table,

                "sys_id": result.get("sys_id"),

                "response_code": response.status_code
            }

        except Exception as e:

            logger.error(f"Create failed: {e}")

            return {

                "status": "error",

                "action": "CREATE_FAILED",

                "name": ci_name,

                "table": table,

                "error": str(e)
            }

# -----------------------------------------------------------
# Bulk UPSERT
# -----------------------------------------------------------

def bulk_insert(

    records,

    delay_between=0.2,

    instance=None,

    username=None,

    password=None
):

    results = []

    success_count = 0

    error_count = 0

    for i, record in enumerate(records):

        table = record.get("snow_table", "cmdb_ci")

        result = upsert_ci(

            record,

            table,

            instance,

            username,

            password
        )

        results.append(result)

        if result["status"] == "success":

            success_count += 1

        else:

            error_count += 1

        logger.info(

            f"Progress: {i+1}/{len(records)} | "
            f"Success={success_count} | "
            f"Errors={error_count}"
        )

        time.sleep(delay_between)

    summary = {

        "total": len(records),

        "success_count": success_count,

        "error_count": error_count,

        "results": results
    }

    logger.info(

        f"\n{'='*60}\n"

        f"Bulk UPSERT Complete\n"

        f"Total: {len(records)}\n"

        f"Success: {success_count}\n"

        f"Errors: {error_count}\n"

        f"{'='*60}"
    )

    return summary

# -----------------------------------------------------------
# Query CMDB
# -----------------------------------------------------------

def query_cmdb(

    table,

    sysparm_query="",

    sysparm_fields="name,ip_address,os,environment,manufacturer",

    sysparm_limit=50,

    instance=None,

    username=None,

    password=None
):

    inst = instance or SN_INSTANCE

    usr = username or SN_USERNAME

    pwd = password or SN_PASSWORD

    session = _build_session(usr, pwd)

    url = f"{inst}/api/now/table/{table}"

    params = {

        "sysparm_query": sysparm_query,

        "sysparm_fields": sysparm_fields,

        "sysparm_limit": sysparm_limit,

        "sysparm_display_value": "true"
    }

    try:

        response = session.get(

            url,

            params=params,

            timeout=DEFAULT_TIMEOUT
        )

        response.raise_for_status()

        results = response.json().get("result", [])

        logger.info(

            f"Retrieved {len(results)} "
            f"records from {table}"
        )

        return results

    except Exception as e:

        logger.error(f"Query error: {e}")

        return []