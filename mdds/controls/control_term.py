from diffrax import ControlTerm


class CustomControlTerm(ControlTerm):

    @staticmethod
    def prod(vf, control):

        prods = []

        for current_vf, current_control in zip(vf, control):

            if isinstance(current_vf, float):
                prods.append(current_vf * current_control)

            elif current_vf is None: # Could make custom MultiTerm that sums over None and append None
                prods.append(0)

            elif len(current_vf.shape) == 1:
                prods.append(current_vf * current_control)

            elif len(current_vf.shape) == 2:
                prods.append(current_vf @ current_control)

            else:# current_vf is None
                prods.append(0)

        return tuple(prods)
