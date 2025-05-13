#!/usr/bin/env python
"""
Registry Verification Tool - Verifies that the PlotRegistry is properly set up
"""
import sys
import os
import importlib
import inspect
import pkgutil

# Add the parent directory to the path to allow importing the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_registry_setup():
    """Check that the PlotRegistry is properly set up"""
    print("===== Registry Verification Tool =====\n")
    
    issues_found = False
    warnings = []
    passed_tests = []
    
    # Step 1: Check if PlotRegistry exists
    print("Checking PlotRegistry...")
    try:
        from src.registry.plot_registry import PlotRegistry
        passed_tests.append("✅ PlotRegistry class exists")
        
        # Check if it's a singleton
        if hasattr(PlotRegistry, 'get_instance'):
            passed_tests.append("✅ PlotRegistry has get_instance method")
            
            # Try to create an instance
            try:
                registry = PlotRegistry.get_instance()
                passed_tests.append("✅ PlotRegistry instance created successfully")
                
                # Check core methods
                required_methods = ['register_signal', 'connect_node_to_signal', 'get_signal', 'reset']
                missing_methods = [method for method in required_methods if not hasattr(registry, method)]
                
                if not missing_methods:
                    passed_tests.append("✅ PlotRegistry has all required methods")
                else:
                    issues_found = True
                    print(f"❌ PlotRegistry is missing required methods: {', '.join(missing_methods)}")
            except Exception as e:
                issues_found = True
                print(f"❌ Error creating PlotRegistry instance: {e}")
        else:
            issues_found = True
            print("❌ PlotRegistry does not implement the singleton pattern (get_instance method missing)")
    except ImportError as e:
        issues_found = True
        print(f"❌ PlotRegistry not found: {e}")
    
    # Step 2: Check PlotRegistryIntegration
    print("\nChecking PlotRegistryIntegration...")
    try:
        from src.registry.plot_registry_integration import PlotRegistryIntegration
        passed_tests.append("✅ PlotRegistryIntegration class exists")
        
        # Check if it's a singleton
        if hasattr(PlotRegistryIntegration, 'get_instance'):
            passed_tests.append("✅ PlotRegistryIntegration has get_instance method")
            
            # Try to create an instance
            try:
                integration = PlotRegistryIntegration.get_instance()
                passed_tests.append("✅ PlotRegistryIntegration instance created successfully")
                
                # Check core methods
                required_methods = ['connect_plot_unit', 'connect_node_to_signal', 'reset']
                missing_methods = [method for method in required_methods if not hasattr(integration, method)]
                
                if not missing_methods:
                    passed_tests.append("✅ PlotRegistryIntegration has all required methods")
                else:
                    issues_found = True
                    print(f"❌ PlotRegistryIntegration is missing required methods: {', '.join(missing_methods)}")
            except Exception as e:
                issues_found = True
                print(f"❌ Error creating PlotRegistryIntegration instance: {e}")
        else:
            issues_found = True
            print("❌ PlotRegistryIntegration does not implement the singleton pattern (get_instance method missing)")
    except ImportError as e:
        issues_found = True
        print(f"❌ PlotRegistryIntegration not found: {e}")
    
    # Step 3: Check PlotUnit
    print("\nChecking PlotUnit...")
    try:
        from src.plot.fixed_plot_unit import PlotUnit
        passed_tests.append("✅ PlotUnit class exists")
        
        # Check if it's a singleton
        if hasattr(PlotUnit, 'get_instance'):
            passed_tests.append("✅ PlotUnit has get_instance method")
            
            # Check core methods
            required_methods = ['start', 'add_signal_data', 'clear_plots']
            
            # Try to create an instance (might not be the best idea as it could start a window)
            try:
                # Just check for method existence without creating an instance
                missing_methods = [method for method in required_methods 
                                  if not hasattr(PlotUnit, method) and not any(method in attr for attr in dir(PlotUnit))]
                
                if not missing_methods:
                    passed_tests.append("✅ PlotUnit appears to have all required methods")
                else:
                    issues_found = True
                    print(f"❌ PlotUnit is missing required methods: {', '.join(missing_methods)}")
            except Exception as e:
                warnings.append(f"⚠️ Could not fully check PlotUnit methods: {e}")
        else:
            issues_found = True
            print("❌ PlotUnit does not implement the singleton pattern (get_instance method missing)")
    except ImportError as e:
        issues_found = True
        print(f"❌ PlotUnit not found: {e}")
    
    # Step 4: Check node registrations
    print("\nChecking node registrations...")
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'comfy'))
        
        # Check for registry nodes in comfy directory
        expected_nodes = [
            'signal_registry_connector.py',
            'registry_plot_node.py',
            'registry_signal_generator.py',
            'registry_synthetic_generator.py'
        ]
        
        comfy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'comfy')
        found_nodes = [f for f in os.listdir(comfy_dir) if f in expected_nodes]
        
        if set(found_nodes) == set(expected_nodes):
            passed_tests.append("✅ All expected registry node files exist")
        else:
            missing = set(expected_nodes) - set(found_nodes)
            if missing:
                warnings.append(f"⚠️ Some registry node files are missing: {', '.join(missing)}")
        
        # Try to import __init__ to check registrations
        try:
            from comfy import __init__ as comfy_init
            
            # Check if NODE_CLASS_MAPPINGS contains registry nodes
            registry_nodes = ['SignalRegistryConnector', 'RegistryPlotNode', 
                            'RegistrySignalGenerator', 'RegistrySyntheticGenerator']
            
            if hasattr(comfy_init, 'NODE_CLASS_MAPPINGS'):
                registered = [node for node in registry_nodes if node in comfy_init.NODE_CLASS_MAPPINGS]
                
                if len(registered) == len(registry_nodes):
                    passed_tests.append("✅ All registry nodes are registered in NODE_CLASS_MAPPINGS")
                else:
                    missing = set(registry_nodes) - set(registered)
                    if missing:
                        warnings.append(f"⚠️ Some registry nodes are not registered: {', '.join(missing)}")
            else:
                warnings.append("⚠️ NODE_CLASS_MAPPINGS not found in __init__.py")
                
            # Check DISPLAY_NAME_MAPPINGS
            if hasattr(comfy_init, 'NODE_DISPLAY_NAME_MAPPINGS'):
                display_registered = [node for node in registry_nodes if node in comfy_init.NODE_DISPLAY_NAME_MAPPINGS]
                
                if len(display_registered) == len(registry_nodes):
                    passed_tests.append("✅ All registry nodes have display names")
                else:
                    missing = set(registry_nodes) - set(display_registered)
                    if missing:
                        warnings.append(f"⚠️ Some registry nodes are missing display names: {', '.join(missing)}")
            else:
                warnings.append("⚠️ NODE_DISPLAY_NAME_MAPPINGS not found in __init__.py")
                
        except ImportError as e:
            warnings.append(f"⚠️ Could not import comfy.__init__ to check node registrations: {e}")
    
    except Exception as e:
        warnings.append(f"⚠️ Error checking node registrations: {e}")
    
    # Step 5: Check for legacy nodes that might conflict
    print("\nChecking for potential conflicts...")
    legacy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'comfy', 'legacy')
    
    if os.path.exists(legacy_dir):
        passed_tests.append("✅ Legacy directory exists")
        
        conflict_files = ['plot_unit_node.py', 'fixed_plot_unit_node.py', 'synthetic_generator.py']
        legacy_files = os.listdir(legacy_dir)
        
        # Check which potential conflict files are properly in legacy
        in_legacy = [f for f in conflict_files if f in legacy_files]
        if len(in_legacy) == len(conflict_files):
            passed_tests.append("✅ All potential conflicting files are in legacy folder")
        else:
            comfy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'comfy')
            active_conflicts = [f for f in conflict_files if f in os.listdir(comfy_dir)]
            
            if active_conflicts:
                warnings.append(f"⚠️ Potential conflicting files in active directory: {', '.join(active_conflicts)}")
    else:
        warnings.append("⚠️ Legacy directory not found, old nodes might conflict with registry system")
    
    # Print summary
    print("\n===== Verification Summary =====")
    
    if passed_tests:
        print("\nPassed Tests:")
        for test in passed_tests:
            print(f"  {test}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    if issues_found:
        print("\n❌ Some issues were found that may prevent the registry system from working correctly.")
        print("   Please check the details above and fix the identified problems.")
    else:
        if warnings:
            print("\n⚠️ Registry setup appears to be correct, but with some warnings.")
            print("   The system may work but could have issues. Consider addressing the warnings.")
        else:
            print("\n✅ Registry setup appears to be correct! All components are properly configured.")
    
    print("\nFor more help with debugging, see docs/debug_signal_registry.md")

if __name__ == "__main__":
    check_registry_setup()
