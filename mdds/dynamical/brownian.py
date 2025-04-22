from diffrax import VirtualBrownianTree, AbstractPath

import equinox as eqx

from jax import random


class VirtualDiagonalBrownianTrees(AbstractPath):

    shapes: tuple
    brownian_paths: list

    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)

    def __init__(self, t0, t1, tol, shapes, key):
        self.shapes = shapes
        self.t0 = t0
        self.t1 = t1

        keys = random.split(key, len(shapes))
        self.brownian_paths = [VirtualBrownianTree(t0=t0, t1=t1,
                                                   tol=tol,
                                                   shape=(shapes[i],),
                                                   key=keys[i]) for i in range(len(shapes))]

    def evaluate(self, t0, t1=None, left: bool = True) :
        return tuple([b.evaluate(t0=t0, t1=t1, left=left) for b in self.brownian_paths])
