import flwr as fl
import torch
import yaml

from yolov5.models.yolo import Model
from yolov5.utils.general import intersect_dicts

device = "cpu"
ROUNDS = 3

def load_model():
    weights = torch.load("yolov5/weights/yolov5n.pt", map_location="cpu")

    hyp = "yolov5/data/hyps/hyp.scratch-low.yaml"

    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    
    model = Model(weights["model"].yaml, ch=3, nc=80, anchors=hyp.get("anchors")).to(device)
    # ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
    # model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    exclude = []  # exclude keys
    csd = weights["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    # model.load_state_dict(weights['state_dict'])
    return model

def update_model(model, new_weights):
    state_dict = model.state_dict() 
    new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), new_weights)}
    model.load_state_dict(new_state_dict)
    return model

def save_model(model, path="new_model.pt"):
    torch.save(model, path)

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_weights is not None and rnd == ROUNDS:
            model = load_model()
            new_weights = fl.common.parameters_to_ndarrays(aggregated_weights.parameters)
            updated_model = update_model(model, new_weights)
            save_model(updated_model, path="new_model.pt")
            print("Final model saved to 'new_model.pt'")
        
        return aggregated_weights


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return { "accuracy": sum(accuracies) / sum(examples) }


if __name__ == "__main__":
    strategy = CustomFedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(load_model())),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8000",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
