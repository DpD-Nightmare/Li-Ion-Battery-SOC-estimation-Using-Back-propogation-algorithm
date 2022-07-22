# Li-ion Battery SOC estimation using BPNN algorithm

The lithium-ion battery is the most suitable choice for electric vehicles (EV) due
to its advantages of high voltage, high energy density, low self-discharge rate, long
lifecycles and its also support fast charging. The healthy and accurate operation of
EV highly depends on the operation of the battery management system(BMS). The
state of charge(SOC) is one of the crucial parameters of BMS which represents the
amount of charge left in the battery. A good and accurate SOC estimation leads to a
long life expectancy of the battery, catastrophe avoidance from battery failure and,
efficient EV operation. However, SOC estimation is a complex process due to its
dependency on various factors such as ambient temperature, battery age, and many
other factors. The main objective of this thesis is to develop an accurate SOC esti-
mation approach for Li-ion battery by an back-propagation neural network (BPNN)
model and, the hyperparameters of the BPNN model are tuned by the grid search
algorithm. The dynamic discharge profile data of Li-ion battery at 25oC was applied
as a training data set for the BPNN model. This method was validated by dynamic
street test(DST), US06 highway driving schedule, and federal urban driving sched-
ule(FUDS) discharge profiles at various temperatures, from 0oC to 50oC interval of
10oC.

