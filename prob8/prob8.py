import numpy as np
import math

def prob_x_given_beta(x,beta):
    return beta*((x-27)**(beta-1))*math.exp(-(x-27)**beta)

def prob_beta(beta):
    return 0.5*(beta**2)*math.exp(-beta)

def main():
    num = prob_x_given_beta(28,2)*prob_x_given_beta(29,2)*prob_beta(2)
    denom = prob_x_given_beta(28,3)*prob_x_given_beta(29,3)*prob_beta(3)

    print(num/denom)


if __name__ == "__main__":
    main()
