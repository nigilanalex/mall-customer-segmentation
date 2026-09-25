"""Capture actual dashboard pages using installed Microsoft Edge.

Optional: pip install -r requirements-dev.txt
Run from project root: python scripts/capture_screenshots.py
Starts a temporary local server and always shuts it down after verification.
"""
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parents[1]
output = BASE / "documentation" / "screenshots"
output.mkdir(parents=True, exist_ok=True)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
url = f"http://127.0.0.1:{port}"
log = (BASE / "screenshot-server.log").open("w")
process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(BASE / "app.py"),
                            "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={port}"],
                           cwd=BASE, stdout=log, stderr=log,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
try:
    for attempt in range(100):
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=1)
            break
        except OSError:
            time.sleep(.2)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto(url)
        page.get_by_role("heading", name="Administrator login").wait_for(timeout=60000)
        page.screenshot(path=str(output / "00_login.png"))
        page.get_by_label("Username", exact=True).fill("admin")
        page.get_by_label("Password", exact=True).fill("wrong-password")
        page.get_by_role("button", name="Sign in", exact=True).click()
        page.get_by_text("Sign-in failed.", exact=False).wait_for()
        page.get_by_label("Username", exact=True).fill("admin")
        page.get_by_label("Password", exact=True).fill("admin123")
        page.get_by_role("button", name="Sign in", exact=True).click()
        page.get_by_role("heading", name="Home dashboard", exact=True).wait_for(timeout=60000)
        page.locator('[data-testid="stPlotlyChart"]').first.wait_for()
        page.screenshot(path=str(output / "01_home.png"))
        pages = [("Customer analysis", "02_dataset"), ("Clustering results", "03_segmentation"),
                 ("Model comparison", "04_comparison"), ("Customer prediction", "05_prediction"),
                 ("Marketing recommendations", "06_marketing")]
        for title, file in pages:
            page.get_by_role("link", name=re.compile(re.escape(title))).click()
            page.get_by_role("heading", name=title, exact=True).wait_for()
            if title == "Customer prediction":
                page.get_by_role("button", name="Predict customer segment", exact=True).click()
                page.get_by_role("heading", name="Why this assignment?", exact=True).wait_for()
            else:
                page.locator('[data-testid="stDataFrame"], [data-testid="stPlotlyChart"]').first.wait_for()
            page.screenshot(path=str(output / f"{file}.png"))
            if title == "Clustering results":
                page.get_by_text("3D customer view", exact=True).click()
                page.locator('.gl-container canvas').first.wait_for(state="visible", timeout=30000)
                page.screenshot(path=str(output / "07_rfm_3d.png"))
            if page.locator('[data-testid="stException"]').count():
                raise RuntimeError(f"Dashboard exception on {title}")
        # Exercise the real upload widget with the bundled synthetic customers.
        page.get_by_text("Data and model settings", exact=True).click()
        page.locator('input[type="file"]').nth(0).set_input_files(str(BASE / "dataset" / "Mall_Customers.csv"))
        page.get_by_role("button", name="Run analysis", exact=True).click()
        page.get_by_text("Source: Mall_Customers.csv", exact=True).wait_for(timeout=60000)
        if page.locator('[data-testid="stException"]').count():
            raise RuntimeError("Dashboard upload failed")
        page.get_by_role("button", name=re.compile("Log out")).click()
        page.get_by_role("heading", name="Administrator login").wait_for()
        page.goto(url + "/customer_prediction")
        page.get_by_role("heading", name="Administrator login").wait_for()
        # Verify a phone-width login and dashboard without horizontal overflow.
        page.set_viewport_size({"width": 390, "height": 844})
        qa = BASE / "documentation" / ".qa"
        qa.mkdir(exist_ok=True)
        page.screenshot(path=str(qa / "mobile_login.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile login overflow"
        page.get_by_label("Username", exact=True).fill("admin")
        page.get_by_label("Password", exact=True).fill("admin123")
        page.get_by_role("button", name="Sign in", exact=True).click()
        page.get_by_role("heading", name="Home dashboard", exact=True).wait_for(timeout=60000)
        page.locator('[data-testid="stPlotlyChart"]').first.wait_for()
        page.screenshot(path=str(qa / "mobile_home.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile dashboard overflow"
        assert not page.locator('[data-testid="stException"]').count()
        (BASE / "screenshots").mkdir(exist_ok=True)
        for screenshot in output.glob("*.png"):
            shutil.copy2(screenshot, BASE / "screenshots" / screenshot.name)
        browser.close()
    result = {"screenshots": len(list(output.glob("*.png"))), "login_failure_and_success": "passed",
              "six_pages": "passed", "prediction": "passed", "upload": "passed", "logout_and_protected_url": "passed",
              "3d_rendering": "passed", "mobile_login_and_home": "passed"}
    (BASE / "documentation" / "browser_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
finally:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log.close()
