"""Verify a built image over real HTTP, with durable audit storage and restart checks."""

import argparse
import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def docker(*args: str, env: dict | None = None) -> str:
    return subprocess.check_output(["docker", *args], env=env, text=True).strip()


def wait_ready(base: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            time.sleep(1)
    raise RuntimeError("Container did not become ready within 90 seconds")


def request(base: str, path: str, payload: dict | None, key: str = "") -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="riskshield-ml:1.1")
    args = parser.parse_args()
    name = "riskshield-verify-" + secrets.token_hex(4)
    volume = name + "-audit"
    env = {**os.environ, "RISKSHIELD_API_KEY": secrets.token_urlsafe(32)}
    key = env["RISKSHIELD_API_KEY"]
    created = False
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            "127.0.0.1::8000",
            "-e",
            "RISKSHIELD_API_KEY",
            "--mount",
            f"type=volume,source={volume},target=/app/state",
            args.image,
            env=env,
        )
        created = True
        binding = json.loads(docker("inspect", "--format", "{{json .NetworkSettings.Ports}}", name))
        base = "http://127.0.0.1:" + binding["8000/tcp"][0]["HostPort"]
        wait_ready(base)
        payload = json.loads(Path("data/sample/transaction.json").read_text())
        assert request(base, "/score", payload)[0] == 401
        status, first = request(base, "/score", payload, key)
        assert status == 200
        assert request(base, "/score", payload, key)[1] == first
        assert request(base, "/score", {**payload, "transaction_amount": 1}, key)[0] == 409
        assert (
            request(
                base,
                "/labels",
                {"transaction_id": payload["transaction_id"], "fraud_label": 0},
                key,
            )[0]
            == 200
        )
        assert len(request(base, "/score/batch", {"transactions": [payload, payload]}, key)[1]) == 2
        docker("restart", name)
        # Docker may assign a different ephemeral host port after a restart.
        binding = json.loads(docker("inspect", "--format", "{{json .NetworkSettings.Ports}}", name))
        base = "http://127.0.0.1:" + binding["8000/tcp"][0]["HostPort"]
        wait_ready(base)
        assert request(base, "/score", payload, key)[1] == first
        uid = docker("exec", name, "id", "-u")
        assert uid != "0"
        # Also exercise the Docker HEALTHCHECK, separate from the external readiness probe.
        for _ in range(45):
            health = docker("inspect", "--format", "{{.State.Health.Status}}", name)
            if health == "healthy":
                break
            time.sleep(1)
        assert health == "healthy"
        report = {
            "image": args.image,
            "image_id": docker("inspect", "--format", "{{.Image}}", name),
            "health": health,
            "runtime_uid": uid,
            "unauthenticated_status": 401,
            "scoring_status": status,
            "conflict_status": 409,
            "restart_preserved_response": True,
            "label_ingestion": "passed",
            "batch": "passed",
            "response": first,
        }
        Path("reports/container_verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if created:
            docker("rm", "-f", name)
        docker("volume", "rm", volume)
