"""Test calibrators functionality."""

import pytest
import torch
import numpy as np
import sys
import os

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from calibrators import MCal, PlattCalibrator, TemperatureScaling


class TestMCal:
    """Test MCal calibrator."""
    
    def setup_method(self):
        """Set up test data."""
        torch.manual_seed(1234)
        
        self.d = 4
        self.N = 1000
        
        # Create synthetic test data similar to the notebook
        self.clean_probs = torch.ones(self.N, self.d)
        self.clean_probs /= self.clean_probs.sum(dim=1, keepdim=True)
        
        self.ablated_probs = self.clean_probs + torch.rand_like(self.clean_probs) + torch.eye(self.d)[0]
        self.ablated_probs /= self.ablated_probs.sum(dim=1, keepdim=True)
    
    def test_initialization(self):
        """Test MCal initialization."""
        mcal = MCal(self.d)
        assert mcal.num_classes == self.d
        assert not mcal.is_fitted
        assert mcal.w.shape == (self.d,)
        assert mcal.b.shape == (self.d,)
    
    def test_fit(self):
        """Test MCal fitting."""
        mcal = MCal(self.d)
        stats = mcal.fit(
            self.ablated_probs, 
            self.clean_probs, 
            verbose=False, 
            kappa=1.0,
            lr=1e-3,
            early_stopping=False, 
            max_steps=100  # Reduced for testing
        )
        
        assert mcal.is_fitted
        assert "loss" in stats
        assert "acc" in stats
        assert len(stats["loss"]) == 100
    
    def test_forward(self):
        """Test MCal forward pass."""
        mcal = MCal(self.d)
        mcal.fit(
            self.ablated_probs, 
            self.clean_probs, 
            verbose=False, 
            max_steps=50
        )
        
        output = mcal(self.ablated_probs)
        
        # Check output shape
        assert output.shape == self.ablated_probs.shape
        
        # Check probabilities sum to 1
        assert torch.allclose(output.sum(dim=1), torch.ones(self.N), atol=1e-5)
        
        # Check probabilities are in valid range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_initialization_with_data(self):
        """Test MCal initialization with data."""
        mcal = MCal(self.d, self.ablated_probs, self.clean_probs)
        assert mcal.is_fitted


class TestPlattCalibrator:
    """Test Platt calibrator."""
    
    def setup_method(self):
        """Set up test data."""
        torch.manual_seed(1234)
        
        self.d = 4
        self.N = 1000
        
        self.clean_probs = torch.ones(self.N, self.d)
        self.clean_probs /= self.clean_probs.sum(dim=1, keepdim=True)
        
        self.ablated_probs = self.clean_probs + torch.rand_like(self.clean_probs) + torch.eye(self.d)[0]
        self.ablated_probs /= self.ablated_probs.sum(dim=1, keepdim=True)
    
    def test_initialization(self):
        """Test Platt calibrator initialization."""
        platt = PlattCalibrator(self.d)
        assert platt.num_classes == self.d
        assert not platt.is_fitted
        assert platt.w.shape == (self.d,)
        assert platt.b.shape == (self.d,)
    
    def test_fit(self):
        """Test Platt calibrator fitting."""
        platt = PlattCalibrator(self.d)
        stats = platt.fit(
            self.ablated_probs, 
            self.clean_probs, 
            lr=1e-3, 
            verbose=False, 
            max_steps=100  # Reduced for testing
        )
        
        assert platt.is_fitted
        assert "loss" in stats
        assert "acc" in stats
        assert len(stats["loss"]) == 100
    
    def test_forward(self):
        """Test Platt calibrator forward pass."""
        platt = PlattCalibrator(self.d)
        platt.fit(
            self.ablated_probs, 
            self.clean_probs, 
            verbose=False, 
            max_steps=50
        )
        
        output = platt(self.ablated_probs)
        
        # Check output shape
        assert output.shape == self.ablated_probs.shape
        
        # Check probabilities sum to 1
        assert torch.allclose(output.sum(dim=1), torch.ones(self.N), atol=1e-5)
        
        # Check probabilities are in valid range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)


class TestTemperatureScaling:
    """Test temperature scaling calibrator."""
    
    def setup_method(self):
        """Set up test data."""
        torch.manual_seed(1234)
        
        self.d = 4
        self.N = 100  # Smaller for temperature scaling
        
        self.clean_probs = torch.ones(self.N, self.d)
        self.clean_probs /= self.clean_probs.sum(dim=1, keepdim=True)
        
        self.ablated_probs = self.clean_probs + 0.1 * torch.rand_like(self.clean_probs)
        self.ablated_probs /= self.ablated_probs.sum(dim=1, keepdim=True)
    
    def test_initialization(self):
        """Test temperature scaling initialization."""
        temp_scaler = TemperatureScaling(self.d)
        assert temp_scaler.num_classes == self.d
        assert not temp_scaler.is_fitted
        assert temp_scaler.temperature.shape == (1,)
    
    def test_fit(self):
        """Test temperature scaling fitting."""
        temp_scaler = TemperatureScaling(self.d)
        stats = temp_scaler.fit(
            self.ablated_probs, 
            self.clean_probs, 
            verbose=False,
            max_steps=10  # LBFGS converges quickly
        )
        
        assert temp_scaler.is_fitted
        assert "loss" in stats
        assert "temperature" in stats
    
    def test_forward(self):
        """Test temperature scaling forward pass."""
        temp_scaler = TemperatureScaling(self.d)
        temp_scaler.fit(
            self.ablated_probs, 
            self.clean_probs, 
            verbose=False
        )
        
        output = temp_scaler(self.ablated_probs)
        
        # Check output shape
        assert output.shape == self.ablated_probs.shape
        
        # Check probabilities sum to 1
        assert torch.allclose(output.sum(dim=1), torch.ones(self.N), atol=1e-5)
        
        # Check probabilities are in valid range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)


class TestUtilityFunctions:
    """Test utility functions from notebooks."""
    
    def setup_method(self):
        """Set up test data."""
        torch.manual_seed(1234)
        self.d = 4
        self.N = 100
        
        self.probs = torch.rand(self.N, self.d)
        self.probs /= self.probs.sum(dim=1, keepdim=True)
    
    def test_get_expectation(self):
        """Test get_expectation function equivalent."""
        # Import utility functions
        from utils.optimization import get_expectation, make_one_hot
        
        one_hot_exp, prob_exp = get_expectation(self.probs)
        
        # Test shapes
        assert one_hot_exp.shape == (self.d,)
        assert prob_exp.shape == (self.d,)
        
        # Test that expectations are valid probabilities
        assert torch.allclose(prob_exp.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(one_hot_exp >= 0)
        assert torch.all(prob_exp >= 0)
    
    def test_make_one_hot(self):
        """Test make_one_hot function."""
        from utils.optimization import make_one_hot
        
        one_hot = make_one_hot(self.probs)
        
        # Test shape
        assert one_hot.shape == self.probs.shape
        
        # Test that each row sums to 1
        assert torch.allclose(one_hot.sum(dim=1), torch.ones(self.N))
        
        # Test that values are 0 or 1
        assert torch.all((one_hot == 0) | (one_hot == 1))


if __name__ == "__main__":
    # Run simple tests
    print("Running MCal tests...")
    
    # Test MCal
    test_mcal = TestMCal()
    test_mcal.setup_method()
    test_mcal.test_initialization()
    test_mcal.test_fit()
    test_mcal.test_forward()
    
    print("MCal tests passed!")
    
    # Test Platt
    test_platt = TestPlattCalibrator()
    test_platt.setup_method()
    test_platt.test_initialization()
    test_platt.test_fit()
    test_platt.test_forward()
    
    print("Platt calibrator tests passed!")
    
    # Test Temperature Scaling
    test_temp = TestTemperatureScaling()
    test_temp.setup_method()
    test_temp.test_initialization()
    test_temp.test_fit()
    test_temp.test_forward()
    
    print("Temperature scaling tests passed!")
    
    print("All tests passed!")