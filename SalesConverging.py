import numpy as np
import pandas as pd
import time
import random as rand
import matplotlib.pyplot as plt

class TSPSolver:
    def __init__(self, df_file, cooling_rate=0.95, n_trials=100, max_bad=500):
        self.df = self.data_prep(df_file)
        self.cooling_rate = cooling_rate
        self.n_trials = n_trials
        self.max_bad = max_bad
        self.num_cities = len(self.df)
        self.distance_cache = {}
        self.current_temp = None
        self.percent_move_swap = 0.7
        
        
    def data_prep(self, df_file):
        df = pd.read_csv(df_file, 
                 sep='\s+',
                 comment='#',            # Ignore lines starting with #
                 names=['Longitude', 'Latitude', 'City'],  # Column names
                 quotechar='"')
        df = df[["Longitude", "Latitude"]]
        df['Initial'] =range(0,len(df))
        return df
    
    def get_distance(self, pair):
        cities = [self.df['Initial'].iloc[pair[0]], self.df['Initial'].iloc[pair[1]]]
        key = tuple(sorted(cities))

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
    
    #method to check change in distance of a swap without having to physically swap and calculate total distance
    def check_new_swap(self, pair):
        s1, s2 = np.sort(pair)
        
        # Optimization: Skip if cities are too far apart in the sequence
        if abs(s2 - s1) > self.num_cities // 4 and s1 != 0 and s2 != self.num_cities - 1:
            return -float('inf')  # Don't make this swap
        
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

    #Method to check change in distance of reversing a section without physically reversing or checking total distance
    def check_new_reverse(self, pair):
        s1, s2 = pair
        
        # Optimization: Only consider reversals of reasonable length
        section_length = (s2 - s1) % self.num_cities + 1
        if section_length > self.num_cities // 3:
            return -float('inf')  # Don't make this reversal
            
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
    
    #Method to check change in distance of moving and inserting a section
    def check_new_move(self, pair, pos):
        s1, s2 = pair
        
        # Optimization: Skip if move distance is too large
        move_distance = (pos - s2) % self.num_cities
        if move_distance > self.num_cities // 3:
            return -float('inf')  # Don't make this move
        
        section_length = (s2 - s1) % self.num_cities + 1
        if section_length > self.num_cities // 4:
            return -float('inf')  # Section too large to move
        
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

    #Get random pair for swapping points
    def random_pair_swap(self):
        # Optimization: Focus on nearby cities
        a = rand.randint(0, self.num_cities-1)
        max_distance = max(self.num_cities // 4, 3)  # Allow some long-distance swaps
        offset = rand.randint(1, max_distance)
        b = (a + offset) % self.num_cities
        return [a, b]

    #Get random section for reversing
    def random_pair_reverse(self):
        # Optimization: Focus on smaller sections
        a = rand.randint(0, self.num_cities-1)
        max_length = max(min(int(self.num_cities * self.percent_move_swap), self.num_cities // 3), 3)
        b = (a + rand.randint(2, max_length)) % self.num_cities
        return [a, b]

    #get random section and move location 
    def random_params_move(self):
        # Optimization: Prefer smaller sections and shorter moves
        a = rand.randint(0, self.num_cities-1)
        max_section = max(min(int(self.num_cities * self.percent_move_swap), self.num_cities // 4), 3)
        section_length = rand.randint(2, max_section)
        b = (a + section_length) % self.num_cities
        
        # Choose nearby position for insertion
        max_move = max(self.num_cities // 3, 2)
        move_distance = rand.randint(1, max_move)
        c = (b + move_distance) % self.num_cities
        
        return [a, b], c

    #Run generation at a temperature
    def run_generation(self):
        # Optimization: Adaptive operator selection
        success_counts = {'Swap': 1, 'Reverse': 1, 'Move': 1}
        total_attempts = 3
        
        passed = 0
        for i in range(self.n_trials):
            # Choose operator based on success rate
            probs = [success_counts[op]/total_attempts for op in ['Swap', 'Reverse', 'Move']]
            t = np.random.choice(['Swap', 'Reverse', 'Move'], p=probs)
            
            if t == 'Swap':
                params = self.random_pair_swap()
                difference = self.check_new_swap(params)
            elif t == 'Reverse':
                params = self.random_pair_reverse()
                difference = self.check_new_reverse(params)
            else:
                params = self.random_params_move()
                difference = self.check_new_move(*params)
            
            if difference > -float('inf'):  # Only consider valid moves
                acceptance_prob = np.exp(-abs(difference) / self.current_temp)
                
                if difference > 0 or rand.random() < acceptance_prob:
                    if t == 'Swap':
                        self.swap_points(params)
                    elif t == 'Reverse':
                        self.reverse_section(params)
                    else:
                        self.move_section(*params)
                    passed = 1
                    success_counts[t] += 1
                    total_attempts += 1
        
        return passed

    #Method to calculate initial temperature where 70% of bad moves are accepted
    def calc_initial_temp(self):
        acceptance_probability = .7
        trials = 10000
        return_types = ['Swap', 'Reverse', 'Move']
        
        total = 0
        num = 0
        
        for i in range(trials):
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
                
            if difference > -float('inf') and difference < 0:
                total += difference
                num += 1
                
        if num == 0:
            raise ValueError("Number of worse moves must be greater than zero.")
            
        avg_move = abs(total/num)
        initial_temperature = -avg_move / np.log(acceptance_probability)
        return initial_temperature

    #Method to swap points
    def swap_points(self, pair):
        self.df.iloc[pair[0]], self.df.iloc[pair[1]] = self.df.iloc[pair[1]].copy(), self.df.iloc[pair[0]].copy()

    #Method to reverse section
    def reverse_section(self, pair):
        if pair[0] > pair[1]:
            length = ((self.num_cities - pair[0]) + pair[1] + 1) // 2
        else:
            length = (pair[1] - pair[0] + 1) // 2

        for i in range(length):
            self.swap_points([(pair[0] + i) % self.num_cities, (pair[1] - i) % self.num_cities])
    
    #Method to move section
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
    
    #Total distance calculation
    def total_distance(self):
        total = 0
        for i in range(self.num_cities):
            total += self.get_distance([i, (i+1)%self.num_cities])
        return total

    #Main method to run the TSP solver
    def solve(self):
        self.current_temp = self.calc_initial_temp()
        rand.seed(time.time())
        temps = []
        distances = []
        count = 0  # number of temps without taking a trial
        
        while self.current_temp > 0:
            reset = self.run_generation()
            
            if reset:
                count = 0
            else:
                count += 1
            temps.append(self.current_temp)
            distances.append(self.total_distance())
            
            self.current_temp *= self.cooling_rate
            self.percent_move_swap *= self.cooling_rate
            if count > self.max_bad:
                break
                
        
        return self.df, temps, distances