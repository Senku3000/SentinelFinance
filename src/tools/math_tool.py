"""Math Tool - Python REPL executor for financial calculations"""

import io
import sys
import traceback
from typing import Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import Field


class MathTool(BaseTool):
    """Tool for performing financial calculations using Python REPL"""
    
    name: str = "math_tool"
    description: str = """
    Use this tool to perform financial calculations. 
    NEVER attempt to do math mentally - always use this tool.
    
    Examples:
    - EMI calculations: Calculate monthly EMI for loan amount, interest rate, tenure
    - SIP calculations: Calculate future value of SIP investments
    - Tax calculations: Calculate income tax, deductions, etc.
    - ROI calculations: Calculate returns, CAGR, etc.
    
    Input should be a Python expression or code block that performs the calculation.
    The tool will execute the code and return the result.
    """
    
    def _run(
        self,
        code: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code for financial calculations
        
        Args:
            code: Python code to execute
            description: Optional description of what the calculation does
            
        Returns:
            Dictionary with result, formula, and metadata
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Restricted execution - only allow safe operations
            # In production, use a more robust sandbox
            # Create restricted globals
            restricted_globals = {
                '__builtins__': {
                    'abs': abs, 'round': round, 'min': min, 'max': max,
                    'sum': sum, 'len': len, 'range': range, 'int': int,
                    'float': float, 'str': str, 'bool': bool, 'list': list,
                    'dict': dict, 'tuple': tuple, 'print': print
                },
                'math': __import__('math'),
            }
            
            # Try importing numpy and pandas if available
            try:
                restricted_globals['numpy'] = __import__('numpy')
                restricted_globals['np'] = restricted_globals['numpy']
            except ImportError:
                pass
                
            try:
                restricted_globals['pandas'] = __import__('pandas')
                restricted_globals['pd'] = restricted_globals['pandas']
            except ImportError:
                pass
            
            # Execute the code - use exec for statements, eval for expressions
            local_vars = {}
            if "=" in code or "\n" in code:
                # It's a statement, use exec
                exec(code, restricted_globals, local_vars)
                # Check both local_vars and restricted_globals for result
                if 'result' in local_vars:
                    result = local_vars['result']
                else:
                    result = restricted_globals.get('result')
            else:
                # It's an expression, use eval
                result = eval(code, restricted_globals, {})
            
            # Get any printed output
            output = captured_output.getvalue()
            
            return {
                "success": True,
                "result": result,
                "formula": code,
                "description": description or "Financial calculation",
                "output": output,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            
            return {
                "success": False,
                "result": None,
                "formula": code,
                "description": description or "Financial calculation",
                "output": captured_output.getvalue(),
                "error": error_msg,
                "traceback": traceback_str
            }
            
        finally:
            sys.stdout = old_stdout
    
    async def _arun(
        self,
        code: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of _run"""
        return self._run(code, description)


def create_math_tool() -> MathTool:
    """Factory function to create MathTool instance"""
    return MathTool()
