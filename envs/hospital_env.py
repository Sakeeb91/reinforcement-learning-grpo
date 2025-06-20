import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
import random


class PatientSeverity(IntEnum):
    STANDARD = 0
    URGENT = 1
    CRITICAL = 2


class PatientDemographic(IntEnum):
    PEDIATRIC = 0    # Age < 18
    ADULT = 1        # Age 18-65
    ELDERLY = 2      # Age > 65


@dataclass
class Patient:
    """Represents a patient in the hospital system."""
    id: int
    age: int
    severity: PatientSeverity
    demographic: PatientDemographic
    arrival_time: int
    wait_time: int = 0
    assigned_resource: Optional[int] = None
    treatment_duration: int = 0
    
    @property
    def group_id(self) -> int:
        """Group assignment for GRPO based on demographics and severity."""
        if self.severity == PatientSeverity.CRITICAL:
            return 3  # Critical patients regardless of age
        else:
            return int(self.demographic)  # Age-based groups: 0, 1, 2


@dataclass
class HospitalResources:
    """Hospital resource state."""
    total_beds: int
    available_beds: int
    total_staff: int
    available_staff: int
    equipment_units: int
    available_equipment: int
    
    @property
    def total_equipment(self) -> int:
        """Alias for compatibility."""
        return self.equipment_units
    
    def can_allocate(self, beds_needed: int = 1, staff_needed: int = 1, equipment_needed: int = 0) -> bool:
        """Check if resources can be allocated."""
        return (self.available_beds >= beds_needed and 
                self.available_staff >= staff_needed and 
                self.available_equipment >= equipment_needed)
    
    def allocate(self, beds: int = 1, staff: int = 1, equipment: int = 0):
        """Allocate resources."""
        self.available_beds -= beds
        self.available_staff -= staff
        self.available_equipment -= equipment
    
    def release(self, beds: int = 1, staff: int = 1, equipment: int = 0):
        """Release resources."""
        self.available_beds = min(self.total_beds, self.available_beds + beds)
        self.available_staff = min(self.total_staff, self.available_staff + staff)
        self.available_equipment = min(self.equipment_units, self.available_equipment + equipment)


class HospitalEnv(gym.Env):
    """
    Hospital Resource Allocation Environment for GRPO.
    
    The agent must allocate limited hospital resources (beds, staff, equipment)
    to incoming patients while ensuring fairness across demographic groups.
    """
    
    def __init__(
        self,
        max_patients_per_step: int = 5,
        total_beds: int = 50,
        total_staff: int = 30,
        total_equipment: int = 20,
        max_queue_size: int = 100,
        episode_length: int = 200,
        demographic_arrival_rates: Optional[Dict[PatientDemographic, float]] = None,
        severity_distribution: Optional[Dict[PatientSeverity, float]] = None
    ):
        super().__init__()
        
        # Environment parameters
        self.max_patients_per_step = max_patients_per_step
        self.max_queue_size = max_queue_size
        self.episode_length = episode_length
        
        # Resource configuration
        self.resources = HospitalResources(
            total_beds=total_beds,
            available_beds=total_beds,
            total_staff=total_staff,
            available_staff=total_staff,
            equipment_units=total_equipment,
            available_equipment=total_equipment
        )
        
        # Patient arrival patterns
        self.demographic_rates = demographic_arrival_rates or {
            PatientDemographic.PEDIATRIC: 0.2,
            PatientDemographic.ADULT: 0.5,
            PatientDemographic.ELDERLY: 0.3
        }
        
        self.severity_distribution = severity_distribution or {
            PatientSeverity.STANDARD: 0.6,
            PatientSeverity.URGENT: 0.3,
            PatientSeverity.CRITICAL: 0.1
        }
        
        # State and action spaces
        # State: [resource_availability, queue_stats, time_of_day, demographic_distribution]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
        
        # Action: For each patient in queue, decide priority (0=wait, 1=treat_now, 2=expedite)
        self.action_space = gym.spaces.Discrete(3)
        
        # Internal state
        self.current_step = 0
        self.patient_queue: List[Patient] = []
        self.active_patients: List[Patient] = []
        self.next_patient_id = 0
        self.total_patients_served = 0
        
        # Metrics tracking
        self.demographic_wait_times = {demo: [] for demo in PatientDemographic}
        self.demographic_service_times = {demo: [] for demo in PatientDemographic}
        self.total_wait_time = 0
        self.patients_treated = {demo: 0 for demo in PatientDemographic}
        
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.current_step = 0
        self.patient_queue.clear()
        self.active_patients.clear()
        self.next_patient_id = 0
        self.total_patients_served = 0
        
        # Reset resources
        self.resources.available_beds = self.resources.total_beds
        self.resources.available_staff = self.resources.total_staff
        self.resources.available_equipment = self.resources.total_equipment
        
        # Reset metrics
        self.demographic_wait_times = {demo: [] for demo in PatientDemographic}
        self.demographic_service_times = {demo: [] for demo in PatientDemographic}
        self.total_wait_time = 0
        self.patients_treated = {demo: 0 for demo in PatientDemographic}
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Generate new patients
        self._generate_patients()
        
        # Process current patients (aging, treatment completion)
        self._process_active_patients()
        
        # Apply action to patient queue
        reward = self._apply_action(action)
        
        # Update wait times
        self._update_wait_times()
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Collect metrics
        info = self._get_info()
        
        return self._get_observation(), reward, done, info
    
    def _generate_patients(self):
        """Generate new patients arriving at the hospital."""
        # Time-based arrival rate (more patients during certain hours)
        hour_of_day = (self.current_step % 24)
        if 8 <= hour_of_day <= 18:  # Busy hours
            arrival_rate = np.random.poisson(2)
        elif 20 <= hour_of_day <= 23 or 0 <= hour_of_day <= 6:  # Night shift
            arrival_rate = np.random.poisson(1)
        else:  # Moderate hours
            arrival_rate = int(np.random.poisson(1.5))
        
        arrival_rate = min(arrival_rate, self.max_patients_per_step)
        
        for _ in range(arrival_rate):
            if len(self.patient_queue) < self.max_queue_size:
                patient = self._create_patient()
                self.patient_queue.append(patient)
    
    def _create_patient(self) -> Patient:
        """Create a new patient with realistic characteristics."""
        # Sample demographic
        demo_probs = list(self.demographic_rates.values())
        demographic = PatientDemographic(np.random.choice(len(demo_probs), p=demo_probs))
        
        # Sample severity
        sev_probs = list(self.severity_distribution.values())
        severity = PatientSeverity(np.random.choice(len(sev_probs), p=sev_probs))
        
        # Generate age based on demographic
        if demographic == PatientDemographic.PEDIATRIC:
            age = random.randint(0, 17)
        elif demographic == PatientDemographic.ADULT:
            age = random.randint(18, 65)
        else:  # ELDERLY
            age = random.randint(66, 95)
        
        # Treatment duration based on severity
        if severity == PatientSeverity.CRITICAL:
            treatment_duration = random.randint(8, 24)  # 8-24 hours
        elif severity == PatientSeverity.URGENT:
            treatment_duration = random.randint(4, 12)  # 4-12 hours
        else:  # STANDARD
            treatment_duration = random.randint(2, 8)   # 2-8 hours
        
        patient = Patient(
            id=self.next_patient_id,
            age=age,
            severity=severity,
            demographic=demographic,
            arrival_time=self.current_step,
            treatment_duration=treatment_duration
        )
        
        self.next_patient_id += 1
        return patient
    
    def _apply_action(self, action: int) -> float:
        """Apply the agent's action and return reward."""
        if not self.patient_queue:
            return 0.0
        
        # Simple action interpretation: select next patient to treat
        if action == 0:  # Wait - no patient treated this step
            return self._calculate_reward()
        
        # Find patient to treat based on action
        if action == 1:  # Treat next patient in queue
            patient_idx = 0
        elif action == 2 and len(self.patient_queue) > 1:  # Expedite - treat most urgent
            # Find most urgent patient
            patient_idx = max(range(len(self.patient_queue)), 
                            key=lambda i: self.patient_queue[i].severity.value)
        else:
            patient_idx = 0
        
        if patient_idx < len(self.patient_queue):
            patient = self.patient_queue.pop(patient_idx)
            
            # Check if resources are available
            beds_needed = 1
            staff_needed = 1 if patient.severity != PatientSeverity.STANDARD else 0
            equipment_needed = 1 if patient.severity == PatientSeverity.CRITICAL else 0
            
            if self.resources.can_allocate(beds_needed, staff_needed, equipment_needed):
                # Allocate resources and start treatment
                self.resources.allocate(beds_needed, staff_needed, equipment_needed)
                patient.assigned_resource = self.current_step
                self.active_patients.append(patient)
                
                # Record demographics
                self.patients_treated[patient.demographic] += 1
                self.demographic_wait_times[patient.demographic].append(patient.wait_time)
            else:
                # Put patient back in queue if no resources
                self.patient_queue.insert(0, patient)
        
        return self._calculate_reward()
    
    def _process_active_patients(self):
        """Process patients currently receiving treatment."""
        completed_patients = []
        
        for patient in self.active_patients:
            if patient.assigned_resource is not None:
                treatment_time = self.current_step - patient.assigned_resource
                
                if treatment_time >= patient.treatment_duration:
                    # Treatment completed - release resources
                    beds_to_release = 1
                    staff_to_release = 1 if patient.severity != PatientSeverity.STANDARD else 0
                    equipment_to_release = 1 if patient.severity == PatientSeverity.CRITICAL else 0
                    
                    self.resources.release(beds_to_release, staff_to_release, equipment_to_release)
                    
                    # Record service time
                    service_time = treatment_time + patient.wait_time
                    self.demographic_service_times[patient.demographic].append(service_time)
                    
                    completed_patients.append(patient)
                    self.total_patients_served += 1
        
        # Remove completed patients
        for patient in completed_patients:
            self.active_patients.remove(patient)
    
    def _update_wait_times(self):
        """Update wait times for patients in queue."""
        for patient in self.patient_queue:
            patient.wait_time = self.current_step - patient.arrival_time
            self.total_wait_time += 1
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on efficiency and fairness."""
        if not self.demographic_wait_times:
            return 0.0
        
        # Efficiency component: negative average wait time
        total_waiting = sum(p.wait_time for p in self.patient_queue)
        efficiency_reward = -total_waiting / max(1, len(self.patient_queue))
        
        # Fairness component: minimize disparity between demographic groups
        group_avg_waits = []
        for demo in PatientDemographic:
            demo_patients = [p for p in self.patient_queue if p.demographic == demo]
            if demo_patients:
                avg_wait = sum(p.wait_time for p in demo_patients) / len(demo_patients)
                group_avg_waits.append(avg_wait)
        
        if len(group_avg_waits) > 1:
            fairness_penalty = -np.std(group_avg_waits)  # Negative std deviation
        else:
            fairness_penalty = 0.0
        
        # Critical patient priority
        critical_patients = [p for p in self.patient_queue if p.severity == PatientSeverity.CRITICAL]
        critical_penalty = -sum(p.wait_time * 2 for p in critical_patients)  # Double penalty for critical waits
        
        # Resource utilization bonus
        bed_utilization = (self.resources.total_beds - self.resources.available_beds) / self.resources.total_beds
        utilization_bonus = bed_utilization * 10
        
        total_reward = efficiency_reward + fairness_penalty + critical_penalty + utilization_bonus
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        obs = np.zeros(20, dtype=np.float32)
        
        # Resource availability (0-3)
        obs[0] = self.resources.available_beds / self.resources.total_beds
        obs[1] = self.resources.available_staff / self.resources.total_staff
        obs[2] = self.resources.available_equipment / self.resources.equipment_units
        obs[3] = len(self.patient_queue) / self.max_queue_size
        
        # Time features (4-7)
        obs[4] = (self.current_step % 24) / 24  # Hour of day
        obs[5] = (self.current_step % 168) / 168  # Day of week
        obs[6] = self.current_step / self.episode_length  # Episode progress
        obs[7] = len(self.active_patients) / self.resources.total_beds  # Active patient ratio
        
        # Queue demographics (8-13)
        for i, demo in enumerate(PatientDemographic):
            demo_patients = [p for p in self.patient_queue if p.demographic == demo]
            obs[8 + i] = len(demo_patients) / max(1, len(self.patient_queue))
            if demo_patients:
                obs[11 + i] = sum(p.wait_time for p in demo_patients) / len(demo_patients) / 24  # Normalized avg wait
        
        # Severity distribution in queue (14-16)
        for i, severity in enumerate(PatientSeverity):
            severity_patients = [p for p in self.patient_queue if p.severity == severity]
            obs[14 + i] = len(severity_patients) / max(1, len(self.patient_queue))
        
        # Performance metrics (17-19)
        if self.total_patients_served > 0:
            obs[17] = self.total_patients_served / max(1, self.current_step)  # Throughput
        
        if any(self.demographic_wait_times.values()):
            all_waits = [wait for waits in self.demographic_wait_times.values() for wait in waits]
            obs[18] = np.mean(all_waits) / 24 if all_waits else 0  # Normalized avg historical wait
            obs[19] = np.std(all_waits) / 24 if len(all_waits) > 1 else 0  # Wait time variance
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state."""
        info = {
            'patients_in_queue': len(self.patient_queue),
            'active_patients': len(self.active_patients),
            'total_served': self.total_patients_served,
            'resource_utilization': {
                'beds': (self.resources.total_beds - self.resources.available_beds) / self.resources.total_beds,
                'staff': (self.resources.total_staff - self.resources.available_staff) / self.resources.total_staff,
                'equipment': (self.resources.equipment_units - self.resources.available_equipment) / self.resources.equipment_units
            },
            'demographic_distribution': {
                demo.name: len([p for p in self.patient_queue if p.demographic == demo])
                for demo in PatientDemographic
            },
            'average_wait_times': {
                demo.name: np.mean(waits) if waits else 0
                for demo, waits in self.demographic_wait_times.items()
            }
        }
        return info
    
    def get_fairness_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive fairness metrics."""
        metrics = {}
        
        # Demographic parity in service rates
        total_treated = sum(self.patients_treated.values())
        if total_treated > 0:
            service_rates = {
                demo.name: count / total_treated 
                for demo, count in self.patients_treated.items()
            }
            metrics['service_rate_parity'] = 1.0 - np.std(list(service_rates.values()))
        
        # Wait time equity
        avg_waits = []
        for demo in PatientDemographic:
            if self.demographic_wait_times[demo]:
                avg_waits.append(np.mean(self.demographic_wait_times[demo]))
        
        if len(avg_waits) > 1:
            metrics['wait_time_equity'] = 1.0 / (1.0 + np.std(avg_waits))
            metrics['max_wait_disparity'] = max(avg_waits) - min(avg_waits)
        
        # Overall fairness score
        fairness_components = [v for k, v in metrics.items() if 'equity' in k or 'parity' in k]
        if fairness_components:
            metrics['overall_fairness'] = np.mean(fairness_components)
        
        return metrics