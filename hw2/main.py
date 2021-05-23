import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# app = FastAPI()


# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: Optional[bool] = None


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}

@hydra.main(config_path='config', config_name='params')
def run(cfg: DictConfig) -> None:
    logging.info(f"OMP_NUM_THREADS: {os.getenv('HOST')}:{os.getenv('PORT')}")
    logging.info(OmegaConf.to_yaml(cfg))
    # X, y = load_wine(return_X_y=True, as_frame=True)


    # model = instantiate(cfg.model)
    # scores = cross_val_score(model, X, y, **cfg.cross_val)
    # logging.info(f'Mean score: {np.mean(scores):.4f}. Std: {np.mean(scores):.4f}')


if __name__ == '__main__':
    run()
