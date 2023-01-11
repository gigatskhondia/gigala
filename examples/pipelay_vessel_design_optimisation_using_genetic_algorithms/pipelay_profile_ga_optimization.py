from deap import base
from deap import creator
from deap import tools

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import elitism

from static_pipelay import static_pipe_lay

# thetaPT = 0...5  # Angle of inclination of firing line from horizontal, [deg]
# LFL = 80...150  # Length of pipe on inclined firing line, [m]
# RB = 100...250   # Radius of over-bend curve  between stinger and straight section of firing line, [m]
# ELPT = 5...15   # Height of Point of Tangent above  Reference Point (sternpost at keel level),[m]
# LPT = 5...20  # Horizontal distance between Point of Tangent and Reference Point, [m]
# ELWL = 3...20  # Elevation of Water Level above Reference Point, [m]
# LMP = 2...10   # Horizontal distance between Reference Point and Marriage Point, [m]
# RS = 100...250  # Stinger radius, [m]
# CL = 40...80  # Chord length of the stinger between the Marriage Point and the
#               # Lift-Off Point at the second from last roller , [m]

BOUNDS_LOW = [0, 80, 100, 5, 5, 3, 2, 100, 40]
BOUNDS_HIGH = [5, 150, 250, 15, 20, 20, 10, 250, 80]

NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic Algorithm constants:
POPULATION_SIZE = 250
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

PENALTY_VALUE = 10.0    # fixed penalty for violating a constraint

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)


# define the hyperparameter attributes individually:
for i in range(NUM_OF_PARAMS):
    # "hyperparameter_0", "hyperparameter_1", ...
    toolbox.register("hyperparameter_" + str(i),
                     random.uniform,
                     BOUNDS_LOW[i],
                     BOUNDS_HIGH[i])


# create a tuple containing an attribute generator for each param searched:
hyperparameters = ()

for i in range(NUM_OF_PARAMS):
    hyperparameters = hyperparameters + (toolbox.__getattribute__("hyperparameter_" + str(i)),)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator",
                 tools.initCycle,
                 creator.Individual,
                 hyperparameters,
                 n=1)


# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation
def piplayStatic(individual):
    # Pipe data
    ODs = 323.9  # Outer diameter of steel pipe, [mm]
    ts = 14.2  # Wall thickness of steel pipe, [mm]
    Es = 207  # Young's modulus of steel, [GPa]
    SMYS = 358  # SMYS for X52 steel, [MPa]
    rho_s = 7850  # Density of steel,[kg⋅m^−3]
    tFBE = 0.5  # Thickness of FBE insulation layer, [mm]
    rhoFBE = 1300  # Density of FBE, [kg⋅m^−3]
    tconc = 50  # Thickness of concrete coating,[mm]
    rho_conc = 2250  # Density of concrete,[kg⋅m^−3]

    # Environmental data
    d = 50  # Water depth, [m]
    rho_sea = 1025  # Density of seawater,[kg⋅m^−3]

    # Pipe Launch Rollers
    mu_roller = 0.1  # Roller friction for pipe on stinger.

    # Lay-Barge Input Data
    thetaPT = individual[0]  # Angle of inclination of firing line from horizontal, [deg]
    LFL = individual[1]  # Length of pipe on inclined firing line, [m]
    RB = individual[2]   # Radius of over-bend curve  between stinger and straight section of firing line, [m]
    ELPT = individual[3]   # Height of Point of Tangent above  Reference Point (sternpost at keel level),[m]
    LPT = individual[4]  # Horizontal distance between Point of Tangent and Reference Point, [m]
    ELWL = individual[5]   # Elevation of Water Level above Reference Point, [m]
    LMP = individual[6]    # Horizontal distance between Reference Point and Marriage Point, [m]
    RS = individual[7]  # Stinger radius, [m]
    CL = individual[8]    # Chord length of the stinger between the Marriage Point and the
                          # Lift-Off Point at the second from last roller , [m]

    Ttens_tonnef, TTS_ratio, TopS_ratio = static_pipe_lay(ODs, ts, Es, SMYS, rho_s, tFBE, rhoFBE, tconc, rho_conc,
                                                          d, rho_sea,
                                                          mu_roller,
                                                          thetaPT, LFL, RB, ELPT, LPT, ELWL, LMP, RS, CL)
    if numpy.isnan(Ttens_tonnef):
        Ttens_tonnef = 50

    if TTS_ratio > 0.6 or TTS_ratio < 0.3 or TopS_ratio > 0.9:
        return Ttens_tonnef*PENALTY_VALUE,

    return Ttens_tonnef,


toolbox.register("evaluate", piplayStatic)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)


    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])

    print()
    print("Double check: ")
    # Pipe data
    ODs = 323.9  # Outer diameter of steel pipe, [mm]
    ts = 14.2  # Wall thickness of steel pipe, [mm]
    Es = 207  # Young's modulus of steel, [GPa]
    SMYS = 358  # SMYS for X52 steel, [MPa]
    rho_s = 7850  # Density of steel,[kg⋅m^−3]
    tFBE = 0.5  # Thickness of FBE insulation layer, [mm]
    rhoFBE = 1300  # Density of FBE, [kg⋅m^−3]
    tconc = 50  # Thickness of concrete coating,[mm]
    rho_conc = 2250  # Density of concrete,[kg⋅m^−3]

    # Environmental data
    d = 50  # Water depth, [m]
    rho_sea = 1025  # Density of seawater,[kg⋅m^−3]

    # Pipe Launch Rollers
    mu_roller = 0.1  # Roller friction for pipe on stinger.

    # Lay-Barge Input Data
    thetaPT = best[0]  # Angle of inclination of firing line from horizontal, [deg]
    LFL = best[1]  # Length of pipe on inclined firing line, [m]
    RB = best[2]  # Radius of over-bend curve  between stinger and straight section of firing line, [m]
    ELPT = best[3]  # Height of Point of Tangent above  Reference Point (sternpost at keel level),[m]
    LPT = best[4]  # Horizontal distance between Point of Tangent and Reference Point, [m]
    ELWL = best[5]  # Elevation of Water Level above Reference Point, [m]
    LMP = best[6]  # Horizontal distance between Reference Point and Marriage Point, [m]
    RS = best[7]  # Stinger radius, [m]
    CL = best[8]  # Chord length of the stinger between the Marriage Point and the
    # Lift-Off Point at the second from last roller , [m]

    Ttens_tonnef, TTS_ratio, TopS_ratio = static_pipe_lay(ODs, ts, Es, SMYS, rho_s, tFBE, rhoFBE, tconc, rho_conc,
                                                          d, rho_sea,
                                                          mu_roller,
                                                          thetaPT, LFL, RB, ELPT, LPT, ELWL, LMP, RS, CL)

    print("Ttens_tonnef: ", Ttens_tonnef, )
    print("TTS_ratio: ", TTS_ratio, " < 0.6")
    print("TopS_ratio: ", TopS_ratio, " < 0.9")

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.savefig("gen.png")
    plt.show()


if __name__ == "__main__":
    main()
