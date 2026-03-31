"""User Vault Tool - JSON-based user financial data storage"""

import json
import threading
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.tools import BaseTool
from pydantic import Field, ConfigDict

from ..config import Config


class UserVaultTool(BaseTool):
    """Tool for managing user financial profile data"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "user_vault_tool"
    description: str = """
    Use this tool to get or update user financial profile information.
    
    User profile includes:
    - Income (monthly/annual)
    - Expenses (monthly)
    - Financial goals (with timelines and amounts)
    - Risk tolerance (conservative/moderate/aggressive)
    - Existing investments
    - Tax details (HRA, deductions, etc.)
    
    Operations:
    - get_profile: Retrieve user's financial profile
    - update_profile: Update specific fields in user profile
    - get_goal: Get details of a specific financial goal
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'lock', threading.Lock())
    
    def _get_vault_file(self, user_id: str) -> Path:
        """Get path to user vault file"""
        return Config.get_user_vault_file(user_id)
    
    def _load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user profile from JSON file"""
        vault_file = self._get_vault_file(user_id)
        
        if not vault_file.exists():
            # Return default profile
            return self._get_default_profile(user_id)
        
        try:
            with self.lock:
                with open(vault_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            return {
                "error": f"Failed to load profile: {str(e)}",
                **self._get_default_profile(user_id)
            }
    
    def _save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        """Save user profile to JSON file"""
        vault_file = self._get_vault_file(user_id)
        vault_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with self.lock:
                with open(vault_file, 'w', encoding='utf-8') as f:
                    json.dump(profile, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False
    
    def _get_default_profile(self, user_id: str) -> Dict[str, Any]:
        """Get empty user profile — no hardcoded defaults."""
        return {
            "user_id": user_id,
            "income": {"monthly": None, "annual": None, "source": None},
            "expenses": {"monthly": None, "breakdown": {}},
            "goals": [],
            "risk_tolerance": None,
            "existing_investments": {},
            "tax_details": {},
            "created_at": None,
            "updated_at": None
        }
    
    def _run(
        self,
        operation: str,
        user_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform vault operations
        
        Args:
            operation: 'get_profile', 'update_profile', 'get_goal', 'add_goal'
            user_id: User identifier
            data: Optional data for update operations
            
        Returns:
            Dictionary with operation result
        """
        try:
            if operation == "get_profile":
                profile = self._load_profile(user_id)
                return {
                    "success": True,
                    "operation": "get_profile",
                    "profile": profile
                }
            
            elif operation == "update_profile":
                if not data:
                    return {
                        "success": False,
                        "error": "Data required for update_profile operation"
                    }
                
                profile = self._load_profile(user_id)
                
                # Update profile fields
                for key, value in data.items():
                    if isinstance(value, dict) and key in profile and isinstance(profile[key], dict):
                        profile[key].update(value)
                    else:
                        profile[key] = value
                
                profile["updated_at"] = datetime.now().isoformat()
                
                if self._save_profile(user_id, profile):
                    return {
                        "success": True,
                        "operation": "update_profile",
                        "profile": profile
                    }
                else:
                    return {
                        "success": False,
                        "error": "Failed to save profile"
                    }
            
            elif operation == "get_goal":
                profile = self._load_profile(user_id)
                goal_name = data.get("goal_name") if data else None
                
                if not goal_name:
                    return {
                        "success": False,
                        "error": "goal_name required for get_goal operation"
                    }
                
                goals = profile.get("goals", [])
                goal = next((g for g in goals if g.get("name") == goal_name), None)
                
                if goal:
                    return {
                        "success": True,
                        "operation": "get_goal",
                        "goal": goal
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Goal '{goal_name}' not found"
                    }
            
            elif operation == "add_goal":
                if not data:
                    return {
                        "success": False,
                        "error": "Goal data required for add_goal operation"
                    }
                
                profile = self._load_profile(user_id)
                
                if "goals" not in profile:
                    profile["goals"] = []
                
                profile["goals"].append(data)
                
                if self._save_profile(user_id, profile):
                    return {
                        "success": True,
                        "operation": "add_goal",
                        "goal": data
                    }
                else:
                    return {
                        "success": False,
                        "error": "Failed to save goal"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}. Supported: get_profile, update_profile, get_goal, add_goal"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": operation
            }
    
    async def _arun(
        self,
        operation: str,
        user_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async version of _run"""
        return self._run(operation, user_id, data)


def create_user_vault_tool() -> UserVaultTool:
    """Factory function to create UserVaultTool instance"""
    return UserVaultTool()
