"""
Locust load test for a Kubernetes-hosted HTTP service.

Usage (against a deployed service):
  locust -f locustfile.py --host http://SERVICE_IP:PORT --users 50 --spawn-rate 5 --run-time 30m --headless

This produces time-varying RPS that drives CPU, memory, latency, and error metrics
collected by Prometheus / metrics-server in the cluster.
"""

from locust import HttpUser, task, between, LoadTestShape
import random


class K8sAppUser(HttpUser):
    wait_time = between(0.2, 1.5)

    @task(5)
    def health(self):
        self.client.get("/health")

    @task(10)
    def get_item(self):
        item_id = random.randint(1, 500)
        self.client.get(f"/api/items/{item_id}")

    @task(3)
    def search(self):
        q = random.choice(["alpha", "beta", "gamma", "delta", "omega"])
        self.client.get(f"/api/search?q={q}")

    @task(2)
    def create(self):
        self.client.post("/api/items", json={"name": f"item-{random.randint(1, 9999)}"})


class DailyShape(LoadTestShape):
    """
    Approximate a daily traffic curve for longer experiments.
    Stages: night low -> morning ramp -> midday peak -> evening decline.
    """

    stages = [
        {"duration": 300, "users": 10, "spawn_rate": 2},
        {"duration": 600, "users": 40, "spawn_rate": 5},
        {"duration": 900, "users": 80, "spawn_rate": 8},
        {"duration": 1200, "users": 50, "spawn_rate": 5},
        {"duration": 1500, "users": 20, "spawn_rate": 3},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
