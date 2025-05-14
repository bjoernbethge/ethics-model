"""
Ethics Model API client for Python.

This module provides a client implementation for interacting with
the Ethics Model API from Python applications.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union

import requests
from requests.exceptions import RequestException


class EthicsModelClient:
    """Client for interacting with the Ethics Model API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logging.getLogger("ethics_model.api.client")
    
    def ping(self) -> bool:
        """
        Check if the API is reachable and healthy.
        
        Returns:
            True if the API is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except RequestException as e:
            self.logger.error(f"Error pinging API: {str(e)}")
            return False
    
    def analyze(
        self, 
        text: str, 
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze text for ethical content.
        
        Args:
            text: Text to analyze
            include_details: Include detailed analysis in the response
            
        Returns:
            Analysis results
            
        Raises:
            RequestException: If the request fails
            ValueError: If the API returns an error response
        """
        payload = {
            "text": text,
            "include_details": include_details
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
    
    def analyze_batch(
        self, 
        texts: List[str], 
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Batch analyze multiple texts.
        
        Args:
            texts: List of texts to analyze
            include_details: Include detailed analysis in the response
            
        Returns:
            Batch analysis results
            
        Raises:
            RequestException: If the request fails
            ValueError: If the API returns an error response
        """
        payload = {
            "texts": texts,
            "include_details": include_details
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze/batch",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
    
    def analyze_async(
        self, 
        text: str, 
        include_details: bool = False,
        poll_interval: float = 1.0,
        max_retries: int = 60
    ) -> Dict[str, Any]:
        """
        Asynchronously analyze text and wait for results.
        
        Args:
            text: Text to analyze
            include_details: Include detailed analysis in the response
            poll_interval: Seconds between status checks
            max_retries: Maximum number of status checks
            
        Returns:
            Analysis results
            
        Raises:
            RequestException: If the request fails
            ValueError: If the API returns an error response
            TimeoutError: If the operation times out
        """
        payload = {
            "text": text,
            "include_details": include_details
        }
        
        try:
            # Start the async task
            response = requests.post(
                f"{self.base_url}/analyze/async",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            task_id = response.json().get("task_id")
            
            # Poll for results
            for _ in range(max_retries):
                time.sleep(poll_interval)
                
                task_response = requests.get(
                    f"{self.base_url}/tasks/{task_id}",
                    timeout=self.timeout
                )
                
                if task_response.status_code != 200:
                    error_detail = task_response.json().get("detail", "Unknown error")
                    raise ValueError(f"API error: {error_detail}")
                
                task_status = task_response.json()
                
                if task_status.get("status") == "completed":
                    return task_status.get("result")
                
                if task_status.get("status") == "failed":
                    error_detail = task_status.get("result", {}).get("error", "Unknown error")
                    raise ValueError(f"Analysis failed: {error_detail}")
            
            raise TimeoutError("Operation timed out")
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
    
    def get_frameworks(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get information about supported moral frameworks.
        
        Returns:
            Dictionary with framework information
        """
        try:
            response = requests.get(
                f"{self.base_url}/frameworks",
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
    
    def get_manipulation_techniques(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get information about detectable manipulation techniques.
        
        Returns:
            Dictionary with manipulation technique information
        """
        try:
            response = requests.get(
                f"{self.base_url}/manipulation-techniques",
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
            
    def visualize(
        self,
        text: str,
        visualization_type: str
    ) -> Dict[str, Any]:
        """
        Get visualization data for the provided text.
        
        Args:
            text: Text to visualize
            visualization_type: Type of visualization (attention, frameworks, manipulation, 
                                framing, dissonance)
            
        Returns:
            Visualization data and Plotly configuration
            
        Raises:
            RequestException: If the request fails
            ValueError: If the API returns an error response
        """
        if visualization_type not in ["attention", "frameworks", "manipulation", "framing", "dissonance"]:
            raise ValueError(f"Invalid visualization type: {visualization_type}")
        
        payload = {
            "text": text,
            "visualization_type": visualization_type
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/visualization/visualize",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
            
    def train(
        self,
        train_texts: List[str],
        ethics_labels: List[float],
        manipulation_labels: List[float],
        validation_split: float = 0.2,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        augment: bool = False,
        checkpoint_name: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start asynchronous training of the ethics model.
        
        Args:
            train_texts: Training texts
            ethics_labels: Ethics scores for training texts
            manipulation_labels: Manipulation scores for training texts
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            checkpoint_name: Name to save the checkpoint as
            
        Returns:
            Task ID for tracking the training status
            
        Raises:
            RequestException: If the request fails
            ValueError: If the API returns an error response
        """
        if len(train_texts) != len(ethics_labels) or len(train_texts) != len(manipulation_labels):
            raise ValueError("Number of texts must match number of ethics and manipulation labels")
        
        payload = {
            "train_texts": train_texts,
            "ethics_labels": ethics_labels,
            "manipulation_labels": manipulation_labels,
            "validation_split": validation_split,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "augment": augment,
            "checkpoint_name": checkpoint_name,
            "model_config": model_config
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/training/train",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()["task_id"]
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
            
    def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """
        Check the status of an training task.
        
        Args:
            task_id: Training task ID
            
        Returns:
            Training status information
            
        Raises:
            RequestException: If the request fails
            ValueError: If the API returns an error response
        """
        try:
            response = requests.get(
                f"{self.base_url}/training/train/{task_id}",
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
            
    def get_training_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the results of a completed training task.
        
        Args:
            task_id: Training task ID
            
        Returns:
            Detailed training results
            
        Raises:
            RequestException: If the request fails
            ValueError: If the API returns an error response
        """
        try:
            response = requests.get(
                f"{self.base_url}/training/train/{task_id}/result",
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                raise ValueError(f"API error: {error_detail}")
            
            return response.json()
        
        except RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = EthicsModelClient()
    
    # Check if API is available
    if not client.ping():
        print("API is not available. Please start the API server.")
        exit(1)
    
    # Analyze text
    try:
        result = client.analyze(
            "Companies should prioritize profit over environmental concerns.",
            include_details=False
        )
        
        print("\nAnalysis Result:")
        print(f"Ethics Score: {result['ethics_score']:.2f}")
        print(f"Manipulation Score: {result['manipulation_score']:.2f}")
        print(f"Dominant Framework: {result['dominant_framework']}")
        print("\nSummary:")
        for key, value in result['summary'].items():
            print(f"  {key}: {value}")
    
    except (RequestException, ValueError) as e:
        print(f"Error: {str(e)}")
