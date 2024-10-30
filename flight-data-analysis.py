from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

#function to convert seconds into "HH:mm:ss" format
#F.col(column) is used to reference each column within seconds_to_time
def seconds_to_time(column):
    return F.concat_ws(":",
                       F.floor(F.col(column) / 3600).cast("string"),
                       (F.floor((F.col(column) % 3600) / 60)).cast("string"),
                       (F.col(column) % 60).cast("string"))

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 1
    # Hint: Calculate scheduled vs actual travel time, then find the largest discrepancies using window functions.
    #calculating the scheduled and actual travel times in seconds and place them in a new column
    #unix_timestamp() converts each datetime string into a timestamp representing the number of seconds
    flights_df = flights_df.withColumn("ScheduledTravelTime", F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture"))
    flights_df = flights_df.withColumn("ActualTravelTime", F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture"))
    
    #calculate the the discrepancy with abs between actual and scheduled travel times.
    flights_df = flights_df.withColumn("Discrepancy", F.abs(flights_df["ActualTravelTime"] - flights_df["ScheduledTravelTime"]))
    
    #join flights_df with carriers_df on carrier code to get the full carrier name. select the needed columns
    #used aliases to differentiate between columns in the joined DataFrames
    flight_carrier = flights_df.alias("flights").join(carriers_df.alias("carriers"), F.col("flights.CarrierCode") == F.col("carriers.CarrierCode"), "left") \
                                                .select("flights.FlightNum", "carriers.CarrierName", "flights.Origin", 
                                                        "flights.Destination", "flights.ScheduledTravelTime", 
                                                        "flights.ActualTravelTime", "flights.Discrepancy", 
                                                        "flights.CarrierCode")

                        
    #sets up a window specification for ranking rows within each carrier group.
    #partition groups the data by carrier, allowing us to calculate the largest discrepancy for each carrier and then order by descending order.
    window_spec = Window.partitionBy("CarrierCode").orderBy(F.desc("Discrepancy"))
    
    #rank flights within each carrier group and filter for the largest discrepancy
    #add a rank column where row_number() assigns a unique rank to each row within the carrier group, based on the discrepancy order defined by window_spec
    #keep the rank of 1 to get the highest discrepancy and remove the rank column
    largest_discrepancy = flight_carrier.withColumn("Rank", F.row_number().over(window_spec)) \
                                        .filter(F.col("Rank") == 1).drop("Rank")
                                        
    # Apply the function to convert travel times to a more proper time format
    largest_discrepancy = largest_discrepancy.withColumn("ScheduledTravelTimeFormatted", seconds_to_time("ScheduledTravelTime")).withColumn("ActualTravelTimeFormatted", seconds_to_time("ActualTravelTime")).withColumn("DiscrepancyFormatted", seconds_to_time("Discrepancy"))
    
    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    largest_discrepancy.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 2
    # Hint: Calculate standard deviation of departure delays, filter airlines with more than 100 flights.
    #make a new column that calculates the departure delay with actual and scheduled
    #compute difference in seconds
    #negative difference means early departure whereas postitve means late departure
    flights_with_delay = flights_df.withColumn("DepartureDelay", F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture"))
    
    delay_stats = (flights_with_delay.groupBy("CarrierCode") #group by carrier code to represent airline
                   .agg(F.count("DepartureDelay").alias("FlightCount"), #count the number of flights per carrier
                        F.round(F.stddev("DepartureDelay"), 2).alias("DelayStdDev") #calculates the standard deviation of DepartureDelay, giving us a measure of consistency
                        ).filter("FlightCount > 100") #only consider airlines with over 100 flights
                   )
    
    consistent_airlines = ( #join the carrrier df to get carrier names. order by the smallest stddev
        delay_stats.join(carriers_df, "CarrierCode", "inner").select("CarrierName", "FlightCount", "DelayStdDev").orderBy("DelayStdDev")
    )
    
    #format the stddev to time format for easier analysis
    consistent_airlines = consistent_airlines.withColumn("DelayStdDevFormat", seconds_to_time("DelayStdDev"))
    
    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    consistent_airlines.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # TODO: Implement the SQL query for Task 3
    # Hint: Calculate cancellation rates for each route, then join with airports to get airport names.
    #add a new column with 1 and 0 if departure is null, meaning canceled. will help with counting canceled flights
    flights_canceled_flag = flights_df.withColumn("IsCanceled?", 
                                                  F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0)
                                                  )
    
    #calculate cancellation rate
    routes_canceled = (
        flights_canceled_flag.groupBy("Origin", "Destination") #group by origin and destination
        .agg(F.count("*").alias("TotalFlightsOnRoute"), #total flights per route
             F.sum("IsCanceled?").alias("TotalCanceledFlights")) #total canceled flights, counted and summed by 1
    )
    
    #calculate cancellation rate divide canceled flights and total flights
    routes_canceled = routes_canceled.withColumn("CancellationRate", F.round((F.col("TotalCanceledFlights") / F.col("TotalFlightsOnRoute")) * 100, 2))

    #joining airport information to get origin and destination names
    #renaming the columns to differentiate between origin and destination fields
    #first join joins the airport_df through airpoirt code and origin code
    #rename the airportcode from airport_df to origin and join through origin in routes_canceled
    #rename the columns in routes_canceled to names of origin
    #second join joins the airport_df through airpoirt code and desintation code
    #rename the airportcode from airport_df to destination and join through destination in routes_canceled
    #have the columns renamed in names of destination
    routes_canceled = routes_canceled \
        .join(airports_df.withColumnRenamed("AirportCode", "Origin"), "Origin") \
        .withColumnRenamed("AirportName", "OriginAirport") \
        .withColumnRenamed("City", "OriginCity") \
        .join(airports_df.withColumnRenamed("AirportCode", "Destination"), "Destination") \
        .withColumnRenamed("AirportName", "DestinationAirport") \
        .withColumnRenamed("City", "DestinationCity")
    
    #select the necessary fields and sort by descening cancellation rate
    canceled_routes = routes_canceled.select(
        "OriginAirport", "OriginCity", "DestinationAirport", "DestinationCity", "CancellationRate").orderBy(F.desc("CancellationRate"))

    
    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    canceled_routes.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # TODO: Implement the SQL query for Task 4
    # Hint: Create time of day groups and calculate average delay for each carrier within each group.
    #create the timeofday column based on the hour of day in the scheduleddeparture
    flights_df = flights_df.withColumn(
        "TimeOfDay",
        F.when((F.hour("ScheduledDeparture") >= 6) & (F.hour("ScheduledDeparture") < 12), "Morning")
        .when((F.hour("ScheduledDeparture") >= 12) & (F.hour("ScheduledDeparture") < 18), "Afternoon")
        .when((F.hour("ScheduledDeparture") >= 18) & (F.hour("ScheduledDeparture") < 24), "Evening")
        .otherwise("Night")
    )
    
    #create column the calculates the delayed time, subtract actual and scheduled
    flights_df = flights_df.withColumn(
        "DepartureDelay",
        F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")
    )
    
    #calculate average departure delay under each time of day and carrier
    avg_delay_day = flights_df.groupBy("CarrierCode", "TimeOfDay").agg(F.round(F.avg("DepartureDelay"), 2).alias("AverageDepartureDelay"))
    
    #make a timeofday partition for average delays
    windows_spec = Window.partitionBy("TimeOfDay").orderBy("AverageDepartureDelay")
    #place a ranking over that partition to rank the carriers
    ranked_df = avg_delay_day.withColumn("Rank", F.row_number().over(windows_spec))
    
    #joing the carriers df
    #select the carrier name, time of day, average delay, and ranking
    #order by time of day and rank
    carrier_performance_time_of_day = ranked_df.join(carriers_df, "CarrierCode") \
                                                .select("CarrierName", "TimeOfDay", "AverageDepartureDelay", "Rank").orderBy("TimeOfDay", "Rank")
    
    #format the average departure delay to a time format for easier analysis on time                                            
    carrier_performance_time_of_day = carrier_performance_time_of_day.withColumn("AverageDepartureDelayFormatted", seconds_to_time("AverageDepartureDelay"))
    
    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    carrier_performance_time_of_day.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()