import json

# Comprehensive list of global professional skills
skills = [
    # Programming & IT
    "python", "javascript", "java", "c++", "c#", "ruby", "go", "php", "typescript", "swift", "kotlin", "rust", "sql", "nosql", "html", "css",
    "react", "angular", "vue", "nodejs", "django", "flask", "spring boot", "ruby on rails", "laravel", "expressjs", "asp.net", "graphql", "rest api",
    "docker", "kubernetes", "aws", "azure", "google cloud", "terraform", "ansible", "jenkins", "git", "github", "ci/cd", "linux", "bash", "powershell",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy", "matplotlib",
    "data analysis", "data engineering", "big data", "hadoop", "spark", "kafka", "tableau", "power bi", "excel", "data visualization", "statistics",
    "cybersecurity", "penetration testing", "cryptography", "firewalls", "siem", "network security", "cloud security", "ethical hacking",
    
    # Engineering & Architecture
    "autocad", "solidworks", "matlab", "revit", "civil 3d", "sketchup", "ansys", "catia", "creo", "sap2000", "etabs", "staad pro", "fusion 360",
    "structural analysis", "fluid mechanics", "thermodynamics", "geotechnical engineering", "surveying", "concrete design", "steel design",
    "project management", "agile", "scrum", "kanban", "prince2", "six sigma", "lean manufacturing", "quality assurance", "logistics", "supply chain",
    "plc programming", "robotics", "embedded systems", "vlsi", "fpga", "microcontrollers", "arduino", "raspberry pi", "iot",
    
    # Business, Marketing, HR
    "seo", "sem", "content marketing", "email marketing", "social media management", "google analytics", "hubspot", "copywriting", "public relations",
    "salesforce", "crm", "b2b sales", "lead generation", "negotiation", "account management", "business development", "customer success",
    "financial modeling", "accounting", "risk management", "forecasting", "valuation", "investment strategy", "bookkeeping", "quickbooks",
    "talent acquisition", "onboarding", "payroll", "employee relations", "benefits administration", "conflict resolution", "workday",
    
    # Healthcare & Science
    "patient care", "medical billing", "hipaa", "emr", "clinical processing", "public health", "anatomy", "physiology", "pharmacology",
    "molecular biology", "genetics", "bioinformatics", "chemistry", "physics", "laboratory techniques", "clinical trials", "data collection",
    
    # Design & Creative
    "photoshop", "illustrator", "indesign", "canva", "premiere pro", "after effects", "figma", "sketch", "adobe xd", "ui/ux", "typography",
    "branding", "visual composition", "video editing", "animation", "3d rendering", "maya", "blender", "cinema 4d", "zbrush"
]

# Ensure uniqueness and lowercasing
formatted_skills = sorted(list(set([s.lower() for s in skills])))

with open('knowledge_graph.json', 'w') as f:
    json.dump(formatted_skills, f, indent=4)

print(f"Knowledge Graph successfully generated with {len(formatted_skills)} critical skills!")
