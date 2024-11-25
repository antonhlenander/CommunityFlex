I_t = 20

C_t = 20 + 30
eta_comp = 0.78

power = pow(I_t, eta_comp)
#upper_term = np.power(I_t, eta_comp)
upper_term = (pow(I_t, eta_comp) - 1)
utility = (upper_term / eta_comp) - C_t
# Final reward
print(upper_term)
print(utility)