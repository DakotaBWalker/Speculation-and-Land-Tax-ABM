CONFIG = 2 #each time we change a setting, add one to this

#simulation
MAX_TIMESTEP = 4 #20
PREF_WEIGHTINGS = ['additive', 'multiplicative']
PREF_WEIGHTING = PREF_WEIGHTINGS[1]

## WORLD ##
SIZE = 100
PR_INIT_DEV = 0.2
SIGMA = 7 
NR_ENVR_ATTR = 0 # number of environmental attractors in the space
NR_ECO_POTENT_SEEDS = 0
NBR_SIZE = 7 #size of the neighborhood for density measures (nbr_size x nbr_size)
RND_OFF_LIMITS = 0.85
LV_DENSITY_EXP = 1.4
LV_HOUSING_EXP = 0.9
LV_MIN = 10000

## AGENTS ##
N_INIT_AGENTS = 300
N_NEW_AGENTS = 30 
DEVELOPER_STARTING_WEALTH_LOG = 16#15.5 #corresponds to $5,389,698
HOMEOWNER_STARTING_WEALTH_LOG = 13.25 #12.75 #corresponds to $344,551
SPECULATOR_STARTING_WEALTH_LOG = 15 #corresponds to $3,269,017
HOMEOWNER_INCOME_RANGE = [5e3, 2e4]
HOMEOWNER_FUTURE_DISCOUNT_RT = 0.1
DEVELOPER_FUTURE_DISCOUNT_RT = 0.25
SPECULATOR_FUTURE_DISCOUNT_RT = 0.33
AG_TYPE_WT = [15, 80, 5] #ordered as developer, homeowner, speculator
DEV_ALTRUISM = [0.1, 0.5]
HOMEOWNER_ALTRUISM = [0.2,1]
SELL_MIN_PREF_DIF = 0.15
DEV_IV_LV_RATIO = 4
CHOICE_PCT = 0.10
LU_SCENARIOS = ['baseline', 'sprawl', 'density']
LU_SCENARIO = LU_SCENARIOS[0]
if LU_SCENARIO == 'baseline':
    NB_PREF_RANGE = [0.01,0.4]
    CBD_PREF_RANGE = [0.2,0.6]
    DENSITY_PREF_RANGE = [0.2,0.4]
    LOTSIZE_PREF_RANGE = [0.01,0.2]
    IDEAL_DENSITY_RANGE = [0.1,0.7]
    IDEAL_LOTSIZE_RANGE = [0.05,1]
elif LU_SCENARIO == 'sprawl':
    NB_PREF_RANGE = [0.3,0.5]
    CBD_PREF_RANGE = [0.1,0.3]
    DENSITY_PREF_RANGE = [0.2,0.5]
    LOTSIZE_PREF_RANGE = [0.2,0.4]
    IDEAL_DENSITY_RANGE = [0.01,0.3]
    IDEAL_LOTSIZE_RANGE = [0.25,1]
elif LU_SCENARIO == 'density':
    NB_PREF_RANGE = [0.01,0.2]
    CBD_PREF_RANGE = [0.5,0.7]
    DENSITY_PREF_RANGE = [0.4,0.6]
    LOTSIZE_PREF_RANGE = [0.05,0.1]
    IDEAL_DENSITY_RANGE = [0.3,0.8]
    IDEAL_LOTSIZE_RANGE = [0.05,0.15]

def set_parameters_for(scenario):
    global NB_PREF_RANGE, CBD_PREF_RANGE, DENSITY_PREF_RANGE, LOTSIZE_PREF_RANGE, IDEAL_DENSITY_RANGE, IDEAL_LOTSIZE_RANGE, LU_SCENARIO
    LU_SCENARIO = scenario
    if LU_SCENARIO == 'baseline':
        NB_PREF_RANGE = [0.01,0.4]
        CBD_PREF_RANGE = [0.2,0.6]
        DENSITY_PREF_RANGE = [0.2,0.4]
        LOTSIZE_PREF_RANGE = [0.01,0.2]
        IDEAL_DENSITY_RANGE = [0.1,0.7]
        IDEAL_LOTSIZE_RANGE = [0.05,1]
    elif LU_SCENARIO == 'sprawl':
        NB_PREF_RANGE = [0.3,0.5]
        CBD_PREF_RANGE = [0.1,0.3]
        DENSITY_PREF_RANGE = [0.2,0.5]
        LOTSIZE_PREF_RANGE = [0.2,0.4]
        IDEAL_DENSITY_RANGE = [0.01,0.3]
        IDEAL_LOTSIZE_RANGE = [0.25,1]
    elif LU_SCENARIO == 'density':
        NB_PREF_RANGE = [0.01,0.2]
        CBD_PREF_RANGE = [0.5,0.7]
        DENSITY_PREF_RANGE = [0.4,0.6]
        LOTSIZE_PREF_RANGE = [0.05,0.1]
        IDEAL_DENSITY_RANGE = [0.3,0.8]
        IDEAL_LOTSIZE_RANGE = [0.05,0.15]

## TAXES & COSTS ##
TAX_SCHEMES = ['SQ', 'LVT', 'ELVT'] #status quo ('SQ'), land value tax ('LVT'), or ecologically-weighted LVT ('ELVT')
TAX_SCHEME = TAX_SCHEMES[1] 
LVT_RATE = 0.1 #rate of land tax
IVT_RATE = 0.035 #rate of improvement tax
UNIT_SF = [1000,2500]
BUILD_COSTS_PSF = [100,200]
AVG_HU_HARD_COST = sum(UNIT_SF)/2 * sum(BUILD_COSTS_PSF)/2
DEV_IV_LV_RATIO = [4,7]
AVG_DEV_IV_LV_RATIO = sum(DEV_IV_LV_RATIO)/2
RESTORATION_COST = 5000 #restoration costs
ECO_BURDEN_DENOM = 200

