import random

class AfricanBuffaloOptimization:
    def __init__(self, num_buffalos=5, max_iter=3):
        self.num_buffalos = num_buffalos
        self.max_iter = max_iter

        self.lr_bounds = (0.0001, 0.01)
        self.conf_bounds = (0.2, 0.6)
        self.iou_bounds = (0.4, 0.7)
        self.batch_choices = [4, 8, 16]

    def initialize(self):
        buffalos = []
        for _ in range(self.num_buffalos):
            buffalos.append({
                "lr": random.uniform(*self.lr_bounds),
                "conf": random.uniform(*self.conf_bounds),
                "iou": random.uniform(*self.iou_bounds),
                "batch": random.choice(self.batch_choices)
            })
        return buffalos

    def fitness(self, metrics):
        return (
            0.6 * metrics["map"] +
            0.2 * metrics["precision"] +
            0.2 * metrics["recall"]
        )

    def optimize(self, evaluate_fn):
        buffalos = self.initialize()
        best_solution = None
        best_score = -1e9

        for iteration in range(self.max_iter):
            print(f"\n[ABO] Iteration {iteration + 1}/{self.max_iter}")

            for buffalo in buffalos:
                metrics = evaluate_fn(buffalo)
                score = self.fitness(metrics)

                if score > best_score:
                    best_score = score
                    best_solution = buffalo.copy()

            # Move buffalos toward best solution
            for buffalo in buffalos:
                buffalo["lr"] = (buffalo["lr"] + best_solution["lr"]) / 2
                buffalo["conf"] = (buffalo["conf"] + best_solution["conf"]) / 2
                buffalo["iou"] = (buffalo["iou"] + best_solution["iou"]) / 2

        return best_solution
