"""
Evaluation and sensitivity analysis for sequential pattern mining.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from ..processors.sequence_generator import CancerSequenceGenerator
from .pattern_mining import SequentialPatternAnalyzer


class SensitivityAnalyzer:
    """Performs sensitivity analysis across different parameter configurations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize sensitivity analyzer.
        
        Args:
            config: Base configuration
        """
        self.base_config = config or {}
        self.results = {}
        
    def analyze_binning_strategies(self, data: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Analyze sensitivity to different binning strategies.
        
        Args:
            data: Input features
            target: Target labels
            
        Returns:
            Results across different binning strategies
        """
        strategies = ['uniform', 'quantile', 'kmeans']
        n_bins_options = [3, 5, 7]
        
        results = {}
        
        for strategy in strategies:
            strategy_results = {}
            
            for n_bins in n_bins_options:
                config = self.base_config.copy()
                config.update({
                    'discretization_strategy': strategy,
                    'n_bins': n_bins,
                    'min_support': 0.1
                })
                
                try:
                    # Generate sequences
                    sequence_generator = CancerSequenceGenerator(config)
                    sequences = sequence_generator.fit_generate(data, target)
                    
                    # Analyze patterns
                    analyzer = SequentialPatternAnalyzer(config)
                    analysis_results = analyzer.analyze(sequences, target.tolist())
                    
                    strategy_results[f'{n_bins}_bins'] = {
                        'total_patterns': analysis_results['summary']['discriminative_patterns'],
                        'unique_malignant': analysis_results['summary']['unique_malignant_patterns'],
                        'unique_benign': analysis_results['summary']['unique_benign_patterns'],
                        'config': config
                    }
                    
                except Exception as e:
                    strategy_results[f'{n_bins}_bins'] = {
                        'error': str(e),
                        'config': config
                    }
            
            results[strategy] = strategy_results
        
        self.results['binning_sensitivity'] = results
        return results
    
    def analyze_support_thresholds(self, data: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Analyze sensitivity to minimum support thresholds.
        
        Args:
            data: Input features
            target: Target labels
            
        Returns:
            Results across different support thresholds
        """
        support_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
        
        results = {}
        
        for min_support in support_thresholds:
            config = self.base_config.copy()
            config.update({
                'min_support': min_support,
                'discretization_strategy': 'quantile',
                'n_bins': 3
            })
            
            try:
                # Generate sequences
                sequence_generator = CancerSequenceGenerator(config)
                sequences = sequence_generator.fit_generate(data, target)
                
                # Analyze patterns
                analyzer = SequentialPatternAnalyzer(config)
                analysis_results = analyzer.analyze(sequences, target.tolist())
                
                results[f'support_{min_support}'] = {
                    'total_patterns_malignant': analysis_results['class_analysis']['malignant']['total_patterns'],
                    'total_patterns_benign': analysis_results['class_analysis']['benign']['total_patterns'],
                    'discriminative_patterns': len(analysis_results['class_comparison'].get('discriminative_patterns', [])),
                    'unique_malignant': len(analysis_results['class_comparison'].get('unique_to_malignant', [])),
                    'unique_benign': len(analysis_results['class_comparison'].get('unique_to_benign', [])),
                    'config': config
                }
                
            except Exception as e:
                results[f'support_{min_support}'] = {
                    'error': str(e),
                    'config': config
                }
        
        self.results['support_sensitivity'] = results
        return results
    
    def analyze_ranking_methods(self, data: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Analyze sensitivity to feature ranking methods.
        
        Args:
            data: Input features
            target: Target labels
            
        Returns:
            Results across different ranking methods
        """
        ranking_methods = ['mutual_info', 'zscore']
        top_k_options = [5, 10, 15, 20]
        
        results = {}
        
        for method in ranking_methods:
            method_results = {}
            
            for top_k in top_k_options:
                config = self.base_config.copy()
                config.update({
                    'ranking_method': method,
                    'top_k': top_k,
                    'discretization_strategy': 'quantile',
                    'n_bins': 3,
                    'min_support': 0.1
                })
                
                try:
                    # Generate sequences
                    sequence_generator = CancerSequenceGenerator(config)
                    sequences = sequence_generator.fit_generate(data, target)
                    
                    # Analyze patterns
                    analyzer = SequentialPatternAnalyzer(config)
                    analysis_results = analyzer.analyze(sequences, target.tolist())
                    
                    method_results[f'top_{top_k}'] = {
                        'avg_sequence_length': np.mean([len(seq) for seq in sequences]),
                        'total_discriminative': len(analysis_results['class_comparison'].get('discriminative_patterns', [])),
                        'unique_malignant': len(analysis_results['class_comparison'].get('unique_to_malignant', [])),
                        'unique_benign': len(analysis_results['class_comparison'].get('unique_to_benign', [])),
                        'config': config
                    }
                    
                except Exception as e:
                    method_results[f'top_{top_k}'] = {
                        'error': str(e),
                        'config': config
                    }
            
            results[method] = method_results
        
        self.results['ranking_sensitivity'] = results
        return results
    
    def get_best_configuration(self) -> Dict[str, Any]:
        """
        Determine the best configuration based on sensitivity analysis.
        
        Returns:
            Best configuration and rationale
        """
        if not self.results:
            raise ValueError("No sensitivity analysis results available.")
        
        best_config = self.base_config.copy()
        rationale = []
        
        # Analyze binning results
        if 'binning_sensitivity' in self.results:
            binning_results = self.results['binning_sensitivity']
            best_strategy = None
            best_score = 0
            
            for strategy, strategy_results in binning_results.items():
                for bins, result in strategy_results.items():
                    if 'error' not in result:
                        score = (result.get('unique_malignant', 0) + 
                               result.get('unique_benign', 0) + 
                               result.get('total_patterns', 0))
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
                            best_config['discretization_strategy'] = strategy
                            best_config['n_bins'] = result['config']['n_bins']
            
            if best_strategy:
                rationale.append(f"Best binning strategy: {best_strategy} with {best_config.get('n_bins', 3)} bins")
        
        # Analyze support results
        if 'support_sensitivity' in self.results:
            support_results = self.results['support_sensitivity']
            best_support = None
            best_discriminative = 0
            
            for support_key, result in support_results.items():
                if 'error' not in result:
                    discriminative_count = result.get('discriminative_patterns', 0)
                    if discriminative_count > best_discriminative:
                        best_discriminative = discriminative_count
                        best_support = result['config']['min_support']
                        best_config['min_support'] = best_support
            
            if best_support:
                rationale.append(f"Best support threshold: {best_support}")
        
        # Analyze ranking results
        if 'ranking_sensitivity' in self.results:
            ranking_results = self.results['ranking_sensitivity']
            best_method = None
            best_total_score = 0
            
            for method, method_results in ranking_results.items():
                for top_k_key, result in method_results.items():
                    if 'error' not in result:
                        score = (result.get('total_discriminative', 0) + 
                               result.get('unique_malignant', 0) + 
                               result.get('unique_benign', 0))
                        if score > best_total_score:
                            best_total_score = score
                            best_method = method
                            best_config['ranking_method'] = method
                            best_config['top_k'] = result['config']['top_k']
            
            if best_method:
                rationale.append(f"Best ranking method: {best_method} with top {best_config.get('top_k', 10)} features")
        
        return {
            'best_config': best_config,
            'rationale': rationale,
            'sensitivity_results': self.results
        }
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive sensitivity analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No sensitivity analysis results available."
        
        report = "# Sensitivity Analysis Report\n\n"
        
        # Binning Strategy Analysis
        if 'binning_sensitivity' in self.results:
            report += "## Binning Strategy Analysis\n\n"
            binning_results = self.results['binning_sensitivity']
            
            for strategy, results in binning_results.items():
                report += f"### {strategy.title()} Strategy\n\n"
                report += "| Bins | Unique Malignant | Unique Benign | Total Patterns |\n"
                report += "|------|------------------|---------------|----------------|\n"
                
                for bins, result in results.items():
                    if 'error' not in result:
                        report += f"| {bins} | {result.get('unique_malignant', 'N/A')} | {result.get('unique_benign', 'N/A')} | {result.get('total_patterns', 'N/A')} |\n"
                    else:
                        report += f"| {bins} | Error | Error | Error |\n"
                
                report += "\n"
        
        # Support Threshold Analysis
        if 'support_sensitivity' in self.results:
            report += "## Support Threshold Analysis\n\n"
            support_results = self.results['support_sensitivity']
            
            report += "| Support | Malignant Patterns | Benign Patterns | Discriminative |\n"
            report += "|---------|-------------------|-----------------|----------------|\n"
            
            for support_key, result in support_results.items():
                if 'error' not in result:
                    support_val = support_key.replace('support_', '')
                    report += f"| {support_val} | {result.get('total_patterns_malignant', 'N/A')} | {result.get('total_patterns_benign', 'N/A')} | {result.get('discriminative_patterns', 'N/A')} |\n"
                else:
                    support_val = support_key.replace('support_', '')
                    report += f"| {support_val} | Error | Error | Error |\n"
            
            report += "\n"
        
        # Ranking Method Analysis
        if 'ranking_sensitivity' in self.results:
            report += "## Ranking Method Analysis\n\n"
            ranking_results = self.results['ranking_sensitivity']
            
            for method, results in ranking_results.items():
                report += f"### {method.title()} Method\n\n"
                report += "| Top K | Avg Seq Length | Discriminative | Unique Malignant | Unique Benign |\n"
                report += "|-------|----------------|----------------|------------------|---------------|\n"
                
                for top_k_key, result in results.items():
                    if 'error' not in result:
                        top_k_val = top_k_key.replace('top_', '')
                        avg_len = f"{result.get('avg_sequence_length', 0):.2f}"
                        report += f"| {top_k_val} | {avg_len} | {result.get('total_discriminative', 'N/A')} | {result.get('unique_malignant', 'N/A')} | {result.get('unique_benign', 'N/A')} |\n"
                    else:
                        top_k_val = top_k_key.replace('top_', '')
                        report += f"| {top_k_val} | Error | Error | Error | Error |\n"
                
                report += "\n"
        
        # Best Configuration
        try:
            best_config_info = self.get_best_configuration()
            report += "## Recommended Configuration\n\n"
            report += f"**Best Configuration:**\n"
            for key, value in best_config_info['best_config'].items():
                report += f"- {key}: {value}\n"
            
            report += f"\n**Rationale:**\n"
            for reason in best_config_info['rationale']:
                report += f"- {reason}\n"
            
        except Exception as e:
            report += f"## Configuration Recommendation\n\nError generating recommendation: {str(e)}\n"
        
        return report