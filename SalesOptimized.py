import numpy as np
import pandas as pd
import time
import random as rand
import matplotlib.pyplot as plt

class TSPSolver:
    def __init__(self, df_file, max_iterations=1000, initial_temp=100, cooling_rate=0.01, n_trials=100):
        self.df = self.data_prep(df_file)
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.n_trials = n_trials
        self.num_cities = len(self.df)
        self.distance_cache = {}
        self.current_temp = initial_temp
        
        
    def data_prep(self, df_file):
        df = pd.read_csv(df_file, 
                 sep='\s+',
                 comment='#',            # Ignore lines starting with #
                 names=['Longitude', 'Latitude', 'City'],  # Column names (some data files may have city names and others don't)
                 quotechar='"')
        df = df[["Longitude", "Latitude"]]
        df['Initial'] =range(0,len(df))
        return df
    
    #get distance between 2 cities
    def get_distance(self, pair):
        cities = [self.df['Initial'].iloc[pair[0]], self.df['Initial'].iloc[pair[1]]]
        key = tuple(sorted(cities))

        # Check if distance is already calculated
        if key in self.distance_cache:
            return self.distance_cache[key]

        s1, s2 = pair
        lat1 = np.radians(self.df["Latitude"].iloc[s1])
        lat2 = np.radians(self.df["Latitude"].iloc[s2])
        long1 = np.radians(self.df["Longitude"].iloc[s1])
        long2 = np.radians(self.df["Longitude"].iloc[s2])
        d_lat = lat1 - lat2
        d_long = long1 - long2
        a = (np.sin(d_lat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(d_long / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = 6371 * c

        self.distance_cache[key] = float(d)
        return float(d)

    #Optimized method to calculate improvement of swapping two cities in the route
    def check_new_swap(self, pair):
        s1, s2 = np.sort(pair)
        
        if s1 == 0 and s2 == self.num_cities - 1:
            s1, s2 = s2, s1

        c_prev1 = (s1 - 1) % self.num_cities
        c_next1 = (s1 + 1) % self.num_cities
        c_prev2 = (s2 - 1) % self.num_cities
        c_next2 = (s2 + 1) % self.num_cities

        if (s1 + 1) % self.num_cities == s2:
            curr_dist = (
                self.get_distance([c_prev1, s1])
                + self.get_distance([s2, c_next2])
            )
            swap_dist = (
                self.get_distance([c_prev1, s2])
                + self.get_distance([s1, c_next2])
            )
        else:
            curr_dist = (
                self.get_distance([c_prev1, s1])
                + self.get_distance([s1, c_next1])
                + self.get_distance([c_prev2, s2])
                + self.get_distance([s2, c_next2])
            )
            swap_dist = (
                self.get_distance([c_prev1, s2])
                + self.get_distance([s2, c_next1])
                + self.get_distance([c_prev2, s1])
                + self.get_distance([s1, c_next2])
            )
        return curr_dist - swap_dist

    #optimized method to calculate distance improvement from reversing a section in the route
    def check_new_reverse(self, pair):
        s1, s2 = pair
        
        c_prev1 = (s1 - 1) % self.num_cities
        c_next2 = (s2 + 1) % self.num_cities
        
        curr_dist = (
            self.get_distance([c_prev1, s1])
            + self.get_distance([s2, c_next2])
        )
        swap_dist = (
            self.get_distance([c_prev1, s2])
            + self.get_distance([s1, c_next2])
        )
        return curr_dist - swap_dist

    #optimized method for calculating change in distance from moving a section of the route
    def check_new_move(self, pair, pos):
        s1, s2 = pair
        
        prev = (s1-1) % self.num_cities
        post = (s2+1) % self.num_cities
        prev_pos = (pos-1) % self.num_cities
        
        curr_dist = (
            self.get_distance([prev, s1])
            + self.get_distance([s2, post])
            + self.get_distance([prev_pos, pos])
        )
        swap_dist = (
            self.get_distance([prev, post])
            + self.get_distance([prev_pos, s1])
            + self.get_distance([s2, pos])
        )
        return curr_dist - swap_dist

    #physically swap two cities in the route
    def swap_points(self, pair):
        self.df.iloc[pair[0]], self.df.iloc[pair[1]] = self.df.iloc[pair[1]].copy(), self.df.iloc[pair[0]].copy()

    #physically reverse a section of the route
    def reverse_section(self, pair):
        if pair[0] > pair[1]:
            length = ((self.num_cities - pair[0]) + pair[1] + 1) // 2
        else:
            length = (pair[1] - pair[0] + 1) // 2

        for i in range(length):
            self.swap_points([(pair[0] + i) % self.num_cities, (pair[1] - i) % self.num_cities])

    #Physically move a section of the route
    def move_section(self, pair, pos):
        start, end = pair

        # Handle wrap-around case and extract the section
        if start <= end:
            section = self.df.iloc[start:end + 1].copy()
            remaining = pd.concat([self.df.iloc[:start], self.df.iloc[end + 1:]]).reset_index(drop=True)
        else:
            section = pd.concat([self.df.iloc[start:], self.df.iloc[:end + 1]]).copy()
            remaining = self.df.iloc[end + 1:start].reset_index(drop=True)

        # Insert the section at the target position
        if pos <= start:
            before = remaining.iloc[:pos]
            after = remaining.iloc[pos:]
            df_new = pd.concat([before, section, after]).reset_index(drop=True)
        else:
            before = remaining.iloc[:pos - len(section)]
            after = remaining.iloc[pos - len(section):]
            df_new = pd.concat([before, section, after]).reset_index(drop=True)

        # Update the DataFrame in place
        self.df.iloc[:] = df_new.iloc[:]

    #calculate total distance of route
    def total_distance(self):
        total = 0
        for i in range(self.num_cities):
            total += self.get_distance([i, (i+1) % self.num_cities])
        return total

    #get two cities to swap
    def random_pair_swap(self):
        a = rand.randint(0, self.num_cities-1)
        b = rand.randint(0, self.num_cities-1)
        while b == a:  # Ensure they are not equal
            b = rand.randint(0, self.num_cities-1)
        return [a, b]

    #get section of route to reverse
    def random_pair_reverse(self):
        a = rand.randint(0, self.num_cities-1)
        b = rand.randint(1, int(self.num_cities**(.8)*2+1))
        if b <= self.num_cities**(.8):
            b = (a+b) % self.num_cities
        else:
            b = (a-b//2) % self.num_cities
            a, b = b, a
        return [a, b]

    #get section of route and location to move it
    def random_params_move(self):
        a = rand.randint(0, self.num_cities-1)
        b = rand.randint(1, int(self.num_cities**(.8)*2+1))
        if b <= self.num_cities**(.8):
            b = (a+b) % self.num_cities
        else:
            b = (a-b//2) % self.num_cities
            a, b = b, a
        length = (b-a) % self.num_cities + 1
        c = (rand.randint(2, self.num_cities-length)+b) % self.num_cities 
        
        return [a, b], c

    #run generation at a temperature
    def run_generation(self):
        best_difference = None
        return_params = None
        return_type = None
        
        return_types = ['Swap', 'Reverse', 'Move']
        
        for i in range(self.n_trials):
            t = return_types[rand.randint(0,2)]
            if t == 'Swap':
                params = self.random_pair_swap()
                difference = self.check_new_swap(params)
            elif t == 'Reverse':
                params = self.random_pair_reverse()
                difference = self.check_new_reverse(params)
            else:
                params = self.random_params_move()
                difference = self.check_new_move(*params)
            
            if return_params is None:
                best_difference = difference
                return_params = params
                return_type = t
                
            elif difference > best_difference:
                best_difference = difference
                return_params = params
                return_type = t
                
        return return_params, return_type, best_difference
    
    
    #Main method to run the TSP solver
    def solve(self):
        self.current_temp = self.initial_temp
        no_improve = 0
        rand.seed(time.time())
        
        temps = []
        distances = []
        
        
        for i in range(self.max_iterations):
            if self.current_temp == 0:
                break
                
            params, method, difference = self.run_generation()
            
            # Probabilistic acceptance
            if params is not None:
                acceptance_prob = np.exp(-abs(difference) / self.current_temp)
                
                if difference > 0 or rand.random() < acceptance_prob:
                    if method == 'Swap':
                        self.swap_points(params)
                    elif method == 'Reverse':
                        self.reverse_section(params)
                    else:
                        self.move_section(*params)
                        
                    if difference > 0:
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            temps.append(self.current_temp)
            distances.append(self.total_distance())
            
            # Cooling schedule
            self.current_temp *= (1-self.cooling_rate)
            
            if no_improve >= self.max_iterations**.5:
                break
        
        
        
        return self.df, temps, distances