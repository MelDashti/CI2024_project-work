import numpy as np

# just an example
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1] / 5)

def f1(x: np.ndarray) ->  np.ndarray: return np.sin(x[0])

def f2(x: np.ndarray) ->  np.ndarray: return ((2.9280*x[0] + x[1] + x[2]) * (x[0] * (24.8006 - 5.4439 * np.cos(1.5377 * (-2.0512 * np.cos(2.1086 / x[0]) - 2.0512 * np.cos(1.3159 * x[0] + 18.8110) + 3.5862) / x[0])) * (2.5198*x[0] + x[1] - 0.2847) * (x[1] * (-x[0] - x[2]) + 2*x[2] + 1) - 2.6333 * (x[2]**3 + x[2] * (-4.0960*x[0] + 0.7524*x[2] - 4.0960 * np.cos(1.1499 * x[2]) - 4.0960 * np.cos(1.3159 * np.cos(35.3119 / x[0]) + 1.3159 * np.cos(1.3159 * x[0] + 0.1497) - 2.3007) + 33.1863) - 0.7524 * np.tan(0.1450 / x[0]) + 0.9116) * np.exp(x[0]) + np.cos(4.3440 / (x[0] * x[2])) + np.cos(1.3159 * x[2]**3 + 0.9901 * x[2] * (-5.4439*x[0] + x[2] - 5.4439 * np.cos(1.1499 * x[2]) - 5.4439 * np.cos(1.3159 * np.cos(34.0852 / x[0]) + 1.3159 * np.cos(2.3007 + 3.1173 / x[2]) - 2.3007) + 43.1071) + 0.9901 * x[2] - 0.9901 * np.tan(0.1450 / x[0]) + 3.5002) + 24096.6003) * (-7.1636 * np.cos(2.1086 / x[0]) - 5.4439 * np.cos(4.3440 / (x[0] * x[2])) - 7.1636 * np.cos(1.3159 * x[0] + 18.8110) - 5.4439 * np.cos(1.3159 * np.cos(1.7682 / x[0]) - 2.2335) + 34.3103))

def f3(x: np.ndarray) -> np.ndarray: return 2*x[0]**2 - x[1]**3 - 3.48004134511183*x[2] + 4.0823630511295

def f4(x: np.ndarray) -> np.ndarray: return -np.sin(x[1] - 1.5174) + np.sin(x[1] + 1.7729) + np.sin(np.cos(x[1]) - np.cos(np.sin(np.exp(0.3925*np.sin(-np.cos(x[1]) + np.cos(np.exp(0.3925*np.sin(-np.cos(x[1]) + np.cos(-np.sin(x[1] - 1.5174) + np.sin(np.sin(np.cos(x[1]))) + 1.9204) + 1.3251)) - np.sin(x[1] - 1.8106) + np.sin(x[1] + np.sin(x[1] + np.sin(-np.cos(x[1]) + np.cos(-np.sin(x[1] - 0.8732) + np.sin(np.sin(np.cos(x[1]))) + 1.4950) + 1.3251)) - 1.7137 + np.exp(-0.1488*np.cos(x[1]))) + np.sin(np.sin(np.cos(x[1]))) + 0.4254) + 1.3251)) + np.sin(x[1] - np.cos(np.sin(x[1] - 1.5174) - 0.4254)) - 0.7616) + 0.4254) + 0.4478) + 2*np.sin(np.cos(x[1])) + 2.6263*np.cos(x[1]) - np.cos(np.sin(0.1488*x[0] - 0.6037)) + np.cos(np.cos(np.sin(0.1567*x[0] - 3.8575))) + 3.5438

def f5(x: np.ndarray) -> np.ndarray: return (np.power(- (x[0] - x[0]), 3) / np.cos(x[0] / x[0])) * (np.power(1.374644653088323 / x[0], 3) * (x[1] * x[1] * (x[0] + x[1])))

def f6(x: np.ndarray) -> np.ndarray: return -0.69953777823026*x[0] + 1.69953777823026*x[1]

def f7(x: np.ndarray) -> np.ndarray: return 1.50709417171588*np.exp(1.13665904347519*x[0]*x[1] - np.cos(5*x[0] - 3*x[1] + 0.53514698244346*np.cos(-3*x[0] + 2.0501325740505 + 1.86864549891329/x[1]) - 0.188826466099849))

def f8(x: np.ndarray) -> np.ndarray: return 89.9207*(1.0*x[5] + 0.4789*np.sin(1.4862*x[5]) + 0.0835*np.cos(1.4862*x[5]) - 0.1641)**3



