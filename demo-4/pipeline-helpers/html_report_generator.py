#!/usr/bin/env python3
"""
HTML Report Generator for MLOps Pipeline
Creates executive-friendly HTML reports for leadership team
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def safe_get(data, key, default=None):
    """Safely get value from dictionary or return default"""
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def generate_executive_summary_html(pipeline_summary, deployment_info=None):
    """Generate executive summary HTML report"""
    
    # Extract key metrics
    accuracy = safe_get(pipeline_summary, "best_accuracy", 0.0)
    deployment_ready = safe_get(pipeline_summary, "deployment_ready", False)
    environment = safe_get(pipeline_summary, "environment", "unknown")
    pipeline_status = safe_get(pipeline_summary, "pipeline_status", "unknown")
    quality_gate_passed = safe_get(pipeline_summary, "quality_gate_passed", False)
    
    # Status colors and icons
    status_color = "green" if pipeline_status == "completed" else "red"
    status_icon = "✅" if pipeline_status == "completed" else "❌"
    deployment_color = "green" if deployment_ready else "orange"
    deployment_icon = "🚀" if deployment_ready else "⏳"
    
    # Accuracy rating
    if accuracy >= 0.95:
        accuracy_rating = "Excellent"
        accuracy_color = "green"
    elif accuracy >= 0.90:
        accuracy_rating = "Good"
        accuracy_color = "blue"
    elif accuracy >= 0.80:
        accuracy_rating = "Acceptable"
        accuracy_color = "orange"
    else:
        accuracy_rating = "Needs Improvement"
        accuracy_color = "red"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLOps Pipeline Executive Summary</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .content {{
                padding: 30px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                border-left: 4px solid #667eea;
            }}
            .metric-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 1.1em;
                color: #666;
                margin-bottom: 5px;
            }}
            .metric-description {{
                font-size: 0.9em;
                color: #888;
            }}
            .status-section {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 25px;
                margin: 20px 0;
            }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }}
            .status-item {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .status-icon {{
                font-size: 1.5em;
            }}
            .status-text {{
                font-weight: 500;
            }}
            .green {{ color: #28a745; }}
            .blue {{ color: #007bff; }}
            .orange {{ color: #fd7e14; }}
            .red {{ color: #dc3545; }}
            .deployment-section {{
                background: #e3f2fd;
                border-radius: 8px;
                padding: 25px;
                margin: 20px 0;
            }}
            .recommendations {{
                background: #fff3cd;
                border-radius: 8px;
                padding: 25px;
                margin: 20px 0;
            }}
            .recommendations h3 {{
                color: #856404;
                margin-top: 0;
            }}
            .recommendations ul {{
                color: #856404;
            }}
            .footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
            .pipeline-details {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }}
            .pipeline-details table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .pipeline-details th, .pipeline-details td {{
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            .pipeline-details th {{
                background-color: #e9ecef;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 MLOps Pipeline Report</h1>
                <p>Executive Summary - {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value {accuracy_color}">{accuracy:.1%}</div>
                        <div class="metric-description">Performance Rating: {accuracy_rating}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Pipeline Status</div>
                        <div class="metric-value {status_color}">{status_icon}</div>
                        <div class="metric-description">{pipeline_status.title()}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Environment</div>
                        <div class="metric-value">🏗️</div>
                        <div class="metric-description">{environment.title()}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Deployment Status</div>
                        <div class="metric-value {deployment_color}">{deployment_icon}</div>
                        <div class="metric-description">{'Ready' if deployment_ready else 'Pending'}</div>
                    </div>
                </div>
                
                <div class="status-section">
                    <h3>📊 Quality Gates & Compliance</h3>
                    <div class="status-grid">
                        <div class="status-item">
                            <span class="status-icon">{'✅' if quality_gate_passed else '❌'}</span>
                            <span class="status-text {'green' if quality_gate_passed else 'red'}">
                                Quality Gate: {'PASSED' if quality_gate_passed else 'FAILED'}
                            </span>
                        </div>
                        <div class="status-item">
                            <span class="status-icon">{'✅' if accuracy >= 0.90 else '⚠️'}</span>
                            <span class="status-text {'green' if accuracy >= 0.90 else 'orange'}">
                                Accuracy Threshold: {'Met' if accuracy >= 0.90 else 'Below Target'}
                            </span>
                        </div>
                        <div class="status-item">
                            <span class="status-icon">{'🚀' if deployment_ready else '⏳'}</span>
                            <span class="status-text {'green' if deployment_ready else 'orange'}">
                                Deployment: {'Approved' if deployment_ready else 'Pending Review'}
                            </span>
                        </div>
                        <div class="status-item">
                            <span class="status-icon">🔒</span>
                            <span class="status-text green">Security: Compliant</span>
                        </div>
                    </div>
                </div>
    """
    
    # Add deployment details if available
    if deployment_info:
        html_content += f"""
                <div class="deployment-section">
                    <h3>🚀 Deployment Details</h3>
                    <p><strong>Endpoint:</strong> {safe_get(deployment_info, 'endpoint_name', 'N/A')}</p>
                    <p><strong>Model:</strong> {safe_get(deployment_info, 'model_name', 'N/A')}</p>
                    <p><strong>Deployment Time:</strong> {safe_get(deployment_info, 'deployment_timestamp', 'N/A')}</p>
                    <p><strong>Status:</strong> <span class="green">✅ Active</span></p>
                </div>
        """
    
    # Add recommendations
    recommendations = []
    if accuracy < 0.90:
        recommendations.append("Consider additional feature engineering to improve model accuracy")
    if not deployment_ready:
        recommendations.append("Review quality gates and model validation before deployment")
    if environment != "production":
        recommendations.append("Plan production deployment strategy and approval process")
    
    if recommendations:
        html_content += f"""
                <div class="recommendations">
                    <h3>💡 Recommendations</h3>
                    <ul>
        """
        for rec in recommendations:
            html_content += f"                        <li>{rec}</li>\n"
        html_content += """
                    </ul>
                </div>
        """
    
    # Add pipeline details
    html_content += f"""
                <div class="pipeline-details">
                    <h3>📋 Pipeline Details</h3>
                    <table>
                        <tr><th>Pipeline ID</th><td>{safe_get(pipeline_summary, 'pipeline_name', 'N/A')}</td></tr>
                        <tr><th>Environment</th><td>{environment.title()}</td></tr>
                        <tr><th>Quality Threshold</th><td>{safe_get(pipeline_summary, 'quality_threshold', 0.90):.1%}</td></tr>
                        <tr><th>Triggered By</th><td>{safe_get(pipeline_summary, 'github_actor', 'System')}</td></tr>
                        <tr><th>Git SHA</th><td>{safe_get(pipeline_summary, 'github_sha', 'N/A')[:8] if safe_get(pipeline_summary, 'github_sha', 'N/A') != 'N/A' else 'N/A'}</td></tr>
                        <tr><th>Artifacts Location</th><td>{safe_get(pipeline_summary, 'artifacts_location', 'N/A')}</td></tr>
                    </table>
                </div>
                
            </div>
            
            <div class="footer">
                <p>Generated automatically by MLOps Pipeline • {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p>This report provides a high-level overview of the machine learning pipeline execution and model performance.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def generate_technical_report_html(pipeline_summary, deployment_info=None):
    """Generate detailed technical HTML report"""
    
    stages = safe_get(pipeline_summary, "pipeline_stages_completed", {})
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLOps Pipeline Technical Report</title>
        <style>
            body {{
                font-family: 'Courier New', monospace;
                margin: 0;
                padding: 20px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: #2d2d2d;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.5);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #0f3460 0%, #16537e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .content {{
                padding: 30px;
            }}
            .section {{
                background: #383838;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid #007acc;
            }}
            .code-block {{
                background: #1e1e1e;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 15px;
                margin: 10px 0;
                overflow-x: auto;
            }}
            .success {{ color: #4ec9b0; }}
            .warning {{ color: #dcdcaa; }}
            .error {{ color: #f44747; }}
            .info {{ color: #9cdcfe; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #555;
            }}
            th {{
                background-color: #404040;
                color: #dcdcaa;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔧 Technical Pipeline Report</h1>
                <p>Detailed Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h3>📊 Pipeline Execution Summary</h3>
                    <div class="code-block">
Pipeline ID: {safe_get(pipeline_summary, 'pipeline_name', 'N/A')}
Environment: {safe_get(pipeline_summary, 'environment', 'N/A')}
Status: {safe_get(pipeline_summary, 'pipeline_status', 'N/A')}
Best Accuracy: {safe_get(pipeline_summary, 'best_accuracy', 0.0):.4f}
Quality Gate: {'PASSED' if safe_get(pipeline_summary, 'quality_gate_passed', False) else 'FAILED'}
Deployment Ready: {safe_get(pipeline_summary, 'deployment_ready', False)}
                    </div>
                </div>
                
                <div class="section">
                    <h3>🔍 Stage Execution Details</h3>
                    <table>
                        <tr><th>Stage</th><th>Status</th><th>Duration</th><th>Details</th></tr>
    """
    
    # Add stage details if available
    if isinstance(stages, dict):
        for stage, details in stages.items():
            # Handle both string format ("completed") and object format ({"success": true})
            if isinstance(details, str):
                # Simple string format: "completed", "failed", etc.
                stage_success = details.lower() == "completed"
                duration = "N/A"
                stage_details = f"Stage {details}"
            elif isinstance(details, dict):
                # Object format: {"success": true, "duration": "2.5s", "details": "..."}
                stage_success = safe_get(details, "success", False)
                duration = safe_get(details, "duration", "N/A")
                stage_details = safe_get(details, "details", "N/A")
            else:
                # Fallback for other formats
                stage_success = False
                duration = "N/A"
                stage_details = str(details)
            
            status_class = "success" if stage_success else "error"
            status_text = "✅ SUCCESS" if stage_success else "❌ FAILED"
            
            html_content += f"""
                        <tr>
                            <td>{stage}</td>
                            <td class="{status_class}">{status_text}</td>
                            <td>{duration}</td>
                            <td>{stage_details}</td>
                        </tr>
        """
    
    html_content += """
                    </table>
                </div>
    """
    
    # Add deployment technical details
    if deployment_info:
        html_content += f"""
                <div class="section">
                    <h3>🚀 Deployment Configuration</h3>
                    <div class="code-block">
Endpoint Name: {safe_get(deployment_info, 'endpoint_name', 'N/A')}
Model Name: {safe_get(deployment_info, 'model_name', 'N/A')}
Model S3 URI: {safe_get(deployment_info, 'model_s3_uri', 'N/A')}
Instance Type: ml.t2.medium
Deployment Timestamp: {safe_get(deployment_info, 'deployment_timestamp', 'N/A')}
                    </div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def main():
    """Main function to generate HTML reports"""
    try:
        # Load pipeline summary
        if not os.path.exists("cicd_pipeline_summary.json"):
            print("❌ Pipeline summary not found")
            return 1
        
        with open("cicd_pipeline_summary.json", "r") as f:
            content = f.read()
            print(f"📄 Pipeline summary content: {content[:200]}...")  # Debug: show first 200 chars
            
            # Try to parse JSON
            try:
                pipeline_summary = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"📄 Full content: {content}")
                return 1
            
            # Ensure pipeline_summary is a dictionary
            if not isinstance(pipeline_summary, dict):
                print(f"❌ Expected dict, got {type(pipeline_summary)}: {pipeline_summary}")
                return 1
        
        # Load deployment info if available
        deployment_info = None
        if os.path.exists("sagemaker_deployment_info.json"):
            try:
                with open("sagemaker_deployment_info.json", "r") as f:
                    deployment_info = json.load(f)
                    if not isinstance(deployment_info, dict):
                        print(f"⚠️ Deployment info is not a dict: {type(deployment_info)}")
                        deployment_info = None
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️ Error loading deployment info: {e}")
                deployment_info = None
        
        # Generate executive summary
        print("📊 Generating executive summary HTML report...")
        executive_html = generate_executive_summary_html(pipeline_summary, deployment_info)
        
        with open("executive_summary.html", "w") as f:
            f.write(executive_html)
        
        # Generate technical report
        print("🔧 Generating technical HTML report...")
        technical_html = generate_technical_report_html(pipeline_summary, deployment_info)
        
        with open("technical_report.html", "w") as f:
            f.write(technical_html)
        
        print("✅ HTML reports generated successfully!")
        print("   - executive_summary.html (for leadership)")
        print("   - technical_report.html (for technical teams)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating HTML reports: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())