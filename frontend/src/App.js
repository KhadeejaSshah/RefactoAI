import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

function App() {
  // Existing state variables
  const [faqStates, setFaqStates] = useState({});
  const [formErrors, setFormErrors] = useState({});
  const [formValues, setFormValues] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  // New state variables for code upload and processing
  const [selectedFile, setSelectedFile] = useState(null);
  const [requirements, setRequirements] = useState('');
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // New state variable for report visibility
  const [showReport, setShowReport] = useState(false);

  // New state variables for generation parameters (using dynamic sliders)
  const [phiMaxLength, setPhiMaxLength] = useState(512);
  const [phiBeamSize, setPhiBeamSize] = useState(5);
  const [phiTemperature, setPhiTemperature] = useState(0.7);
  const [deepseekMaxLength, setDeepseekMaxLength] = useState(6500);
  const [deepseekBeamSize, setDeepseekBeamSize] = useState(1);
  const [deepseekTemperature, setDeepseekTemperature] = useState(0.0);

  // New state for showing the parameters guide modal
  const [showGuide, setShowGuide] = useState(false);

  const toggleFaq = (faqId) => {
    setFaqStates((prevStates) => ({
      ...prevStates,
      [faqId]: !prevStates[faqId],
    }));
  };

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validateForm = () => {
    const errors = {};
    if (!formValues.name.trim()) {
      errors.name = 'Name is required.';
    }
    if (!formValues.email.trim()) {
      errors.email = 'Email is required.';
    } else if (!validateEmail(formValues.email)) {
      errors.email = 'Please enter a valid email address.';
    }
    if (!formValues.subject.trim()) {
      errors.subject = 'Subject is required.';
    } else if (formValues.subject.trim().length < 5) {
      errors.subject = 'Subject must be at least 5 characters long.';
    }
    if (!formValues.message.trim()) {
      errors.message = 'Message is required.';
    } else if (formValues.message.trim().length < 10) {
      errors.message = 'Message must be at least 10 characters long.';
    }
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prevValues) => ({
      ...prevValues,
      [name]: value,
    }));
    setFormErrors((prevErrors) => ({
      ...prevErrors,
      [name]: '',
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      alert('Your message has been successfully submitted!');
      setFormValues({
        name: '',
        email: '',
        subject: '',
        message: '',
      });
      setFormErrors({});
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleRequirementsChange = (e) => {
    setRequirements(e.target.value);
  };

  // Render structured diff from backend (array of {text, added})
  const renderDiff = (diff) => {
    return diff.map((lineObj, index) => {
      if (lineObj.added) {
        return (
          <div key={index} className="added-line">
            {lineObj.text}
          </div>
        );
      } else {
        return (
          <div key={index}>
            {lineObj.text}
          </div>
        );
      }
    });
  };

  const handleProcess = async () => {
    if (!selectedFile) {
      setError('Please upload a Python file.');
      return;
    }
    if (!requirements.trim()) {
      setError('Please enter your requirements.');
      return;
    }
    setError('');
    setProcessing(true);
    setResult(null);
    setShowReport(false);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('requirements', requirements);
    formData.append('phi_max_length', phiMaxLength);
    formData.append('phi_beam_size', phiBeamSize);
    formData.append('phi_temperature', phiTemperature);
    formData.append('deepseek_max_length', deepseekMaxLength);
    formData.append('deepseek_beam_size', deepseekBeamSize);
    formData.append('deepseek_temperature', deepseekTemperature);

    try {
      const response = await axios.post('/analyze-and-modify/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'An error occurred while processing your request.');
    } finally {
      setProcessing(false);
    }
  };

  const faqs = [
    { id: 1, question: 'Is RefactoAI suitable for beginners?', answer: 'Absolutely! RefactoAI provides easy-to-use tools and intuitive guidance, making it accessible for both beginners and experienced developers.' },
    { id: 2, question: 'How does the chatbot assist users?', answer: 'The chatbot provides real-time assistance for coding tasks, requirements gathering, and answers to your technical questions.' },
    { id: 3, question: 'What happens if my code has errors?', answer: 'RefactoAI automatically detects errors in your code and suggests corrections, ensuring it’s error-free and optimized.' },
    { id: 4, question: 'Do I need to install anything to use RefactoAI?', answer: 'No installation is required. RefactoAI is entirely web-based, so you can use it directly from your browser.' },
    { id: 5, question: 'What programming languages are supported?', answer: 'Currently, we support Python, but more languages are on the way! Stay tuned for updates.' },
  ];

  return (
    <div className="App">
      {/* Loading overlay when processing is true */}
      {processing && (
        <div className="loading-overlay">
          <img src="amg3.gif" alt="Loading..." className="loading-gif" />
        </div>
      )}

      {/* Modal for Parameters Guide */}
      {showGuide && (
        <div className="modal-overlay" onClick={() => setShowGuide(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-modal-button" onClick={() => setShowGuide(false)}>Close</button>
            <h3>Parameters Guide</h3>
            <p><strong>User Stories Parameters:</strong></p>
            <ul>
              <li><strong>Max Length:</strong> Determines how many tokens the generated user stories can have. Higher values yield more detailed output but may lead to verbosity.</li>
              <li><strong>Beam Size:</strong> Controls the breadth of search during generation. A larger beam size can improve quality but may slow processing.</li>
              <li><strong>Temperature:</strong> Adjusts randomness. Lower values create more predictable output, while higher values increase variability.</li>
            </ul>
            <p><strong>Code Modification Parameters:</strong></p>
            <ul>
              <li><strong>Max Length:</strong> Sets the maximum token count for the modified code. Ensure this value is high enough to capture all changes.</li>
              <li><strong>Beam Size:</strong> Influences the exploration during code modification. Higher values might improve results at the cost of speed.</li>
              <li><strong>Temperature:</strong> Balances determinism and creativity. Lower temperatures lead to precise modifications; higher ones allow more variation.</li>
            </ul>
          </div>
        </div>
      )}

      <div className="glass-container">
        <div className="navbar">
          <div className="logo">
            Refacto<span style={{ fontWeight: 'lighter', color: '#000' }}>AI</span>
          </div>
          <ul>
            <li>Home</li>
            <li>Chatbot</li>
            <li>Documentation</li>
          </ul>
          <div className="auth-buttons">
            <button className="login-button">Login</button>
            <button className="signup-button">Sign Up</button>
          </div>
        </div>
        <div className="content">
          <div className="text-content">
            <h1>RefactoAI: AI-Driven Code Refactoring & Optimization</h1>
            <p>
              Using our AI-powered software, elevate your coding experience with seamless user
              requirement generation, refactoring code, automated testing, and tailored code
              solutions—effortlessly in one platform.
            </p>
            <button className="get-started-button">Get Started</button>
          </div>
          <div className="image-content">
            <img src="home_girl.png" alt="Illustration of AI-driven code optimization" />
          </div>
        </div>
      </div>

      <div className="features-section">
        <header className="features-header">
          <h1>Explore Our Features</h1>
        </header>
        <div className="features-container">
          <div className="feature-card">
            <h2>AI-Powered Code Refactoring</h2>
            <p>
              Transform your Python code into clean and efficient scripts. Powered by cutting-edge AI,
              RefactoAI ensures your code is refactored with precision for maximum maintainability.
            </p>
          </div>
          <div className="feature-card">
            <h2>Automated Code Analysis</h2>
            <p>
              Stay ahead with intelligent error detection and code quality analysis. Our system
              highlights inefficiencies and provides actionable improvements, making your code
              flawless and future-ready.
            </p>
          </div>
          <div className="feature-card">
            <h2>User Stories Generation</h2>
            <p>
              Effortlessly turn your requirements into actionable user stories. RefactoAI leverages
              advanced LLMs to create tailored user stories that simplify your development process
              and accelerate project execution.
            </p>
          </div>
          <div className="feature-card">
            <h2>Interactive Chatbot Assistance</h2>
            <p>
              Get real-time support from our AI chatbot. Whether it's refactoring your code,
              understanding user requirements, generating user stories, or analyzing code, RefactoAI
              is here to assist you every step of the way.
            </p>
          </div>
        </div>
      </div>

      <section className="code-analysis-section">
        <div className="code-analysis-container">
          <h2>Analyze and Modify Your Python Code</h2>
          <p>
            Upload your Python file and specify your requirements along with generation parameters to get a modified version of your code.
          </p>

          <div className="form-group">
            <label htmlFor="file-upload">Upload Python File:</label>
            <input type="file" id="file-upload" accept=".py" onChange={handleFileChange} />
          </div>

          <div className="form-group">
            <label htmlFor="requirements">Enter Your Requirements:</label>
            <textarea
              id="requirements"
              placeholder="Describe the features or refactoring you need..."
              value={requirements}
              onChange={handleRequirementsChange}
            ></textarea>
          </div>

          <div className="generation-parameters">
            <h3>Adjust Generation Parameters</h3>
            <button className="params-guide-button" onClick={() => setShowGuide(true)}>
              Parameters Guide
            </button>
            <div className="parameter-group">
              <h4>User Stories</h4>
              <label>
                Max Length: {phiMaxLength}
                <input
                  type="range"
                  min="100"
                  max="1024"
                  step="1"
                  value={phiMaxLength}
                  className="slider"
                  onChange={(e) => setPhiMaxLength(e.target.value)}
                />
              </label>
              <label>
                Beam Size: {phiBeamSize}
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={phiBeamSize}
                  className="slider"
                  onChange={(e) => setPhiBeamSize(e.target.value)}
                />
              </label>
              <label>
                Temperature: {phiTemperature}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={phiTemperature}
                  className="slider"
                  onChange={(e) => setPhiTemperature(e.target.value)}
                />
              </label>
            </div>
            <div className="parameter-group">
              <h4>Code Modification</h4>
              <label>
                Max Length: {deepseekMaxLength}
                <input
                  type="range"
                  min="1000"
                  max="10000"
                  step="100"
                  value={deepseekMaxLength}
                  className="slider"
                  onChange={(e) => setDeepseekMaxLength(e.target.value)}
                />
              </label>
              <label>
                Beam Size: {deepseekBeamSize}
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={deepseekBeamSize}
                  className="slider"
                  onChange={(e) => setDeepseekBeamSize(e.target.value)}
                />
              </label>
              <label>
                Temperature: {deepseekTemperature}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={deepseekTemperature}
                  className="slider"
                  onChange={(e) => setDeepseekTemperature(e.target.value)}
                />
              </label>
            </div>
          </div>

          {error && <p className="error-message">{error}</p>}

          <button onClick={handleProcess} disabled={processing}>
            {processing ? 'Processing...' : 'Process Code'}
          </button>

          {result && (
            <div className="result-section">
              {result.syntax.errors && (
                <div className="error-section">
                  <h3>Syntax Errors:</h3>
                  <pre>{result.syntax.errors}</pre>
                </div>
              )}
              {result.pep8.violations && (
                <div className="error-section">
                  <h3>PEP8 Violations:</h3>
                  <pre>{result.pep8.violations}</pre>
                </div>
              )}
              <div className="code-quality-section">
                <h3>Code Quality Analysis:</h3>
                <pre>{result.code_quality.complexity}</pre>
                <pre>{result.code_quality.maintainability}</pre>
              </div>
              <h3 className="section-heading">User Stories:</h3>
              <pre>{result.user_stories}</pre>
              <h3 className="section-heading">Modified Code:</h3>
              <pre>{renderDiff(result.diff)}</pre>

              <button
                onClick={() => setShowReport(!showReport)}
                className="generate-report-button"
              >
                {showReport ? 'Hide Report' : 'Generate Report'}
              </button>

              {showReport && result.report && (
                <div className="report-section">
                  <h3 className="section-heading">Requirement Report:</h3>
                  <table className="report-table">
                    <thead>
                      <tr>
                        <th>User Story ID</th>
                        <th>User Story</th>
                        <th>Requirement Met</th>
                        <th>Matched Elements</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.report).map(([id, reportItem]) => (
                        <tr key={id}>
                          <td>{id}</td>
                          <td>{reportItem.user_story}</td>
                          <td>{reportItem.requirement_met}</td>
                          <td>
                            {Array.isArray(reportItem.matched_elements)
                              ? reportItem.matched_elements.join(', ')
                              : reportItem.matched_elements}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <div className="how-to-use-section">
        <div className="how-to-use-container">
          <header className="how-to-use-header">
            <h1>How to Use RefactoAI</h1>
          </header>
          <div className="steps-container">
            <div className="step-card-container">
              <img src="upload_image.png" alt="Upload Example" className="step-image step-image-left" />
              <div className="step-card">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h2>Upload Your Code File</h2>
                  <p>
                    Drag & drop or upload your Python file. Your journey to cleaner, optimized code begins here!
                  </p>
                </div>
              </div>
            </div>
            <div className="step-card-container">
              <div className="step-card">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h2>Let AI Analyze Your Code</h2>
                  <p>
                    RefactoAI runs a deep analysis to check for issues and gather insights. It’s like having a code wizard inspect every detail!
                  </p>
                </div>
              </div>
              <img src="user_req_1.png" alt="Analyze Example" className="step-image step-image-right" />
            </div>
            <div className="step-card-container">
              <img src="refactor.png" alt="Requirements Example" className="step-image step-image-left" />
              <div className="step-card">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h2>Tell Us What You Need</h2>
                  <p>
                    Enter your requirements—refactor for speed, readability, or specific tasks. We make it exactly how you want!
                  </p>
                </div>
              </div>
            </div>
            <div className="step-card-container">
              <div className="step-card">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h2>Watch the Magic Happen!</h2>
                  <p>
                    RefactoAI transforms your code with precision and intelligence. Enjoy clean, optimized, and efficient code in no time.
                  </p>
                </div>
              </div>
              <img src="code.png" alt="Magic Example" className="step-image step-image-right" />
            </div>
          </div>
        </div>
      </div>

      <section className="contact-us-section">
        <div className="contact-us-container">
          <div className="contact-info">
            <h2>Contact Information</h2>
            <p>We’re here to assist you with RefactoAI. Feel free to reach out anytime!</p>
            <ul>
              <li>+123-456-7890</li>
              <li>support@refactoai.com</li>
              <li>123 Islamabad, Pakistan</li>
            </ul>
          </div>
          <div className="contact-form">
            <h2>Get In Touch</h2>
            <p>Have questions or need help? Send us a message, and we’ll get back to you as soon as possible.</p>
            <form onSubmit={handleSubmit}>
              <div className="form-row">
                <input
                  type="text"
                  name="name"
                  placeholder="Your Name"
                  value={formValues.name}
                  onChange={handleInputChange}
                  className={formErrors.name ? 'error' : ''}
                />
                {formErrors.name && <p className="error-message">{formErrors.name}</p>}
                <input
                  type="email"
                  name="email"
                  placeholder="Your Email"
                  value={formValues.email}
                  onChange={handleInputChange}
                  className={formErrors.email ? 'error' : ''}
                />
                {formErrors.email && <p className="error-message">{formErrors.email}</p>}
              </div>
              <div className="form-row">
                <input
                  type="text"
                  name="subject"
                  placeholder="Your Subject"
                  value={formValues.subject}
                  onChange={handleInputChange}
                  className={formErrors.subject ? 'error' : ''}
                />
                {formErrors.subject && <p className="error-message">{formErrors.subject}</p>}
              </div>
              <div className="form-row">
                <textarea
                  name="message"
                  placeholder="Write your message here..."
                  value={formValues.message}
                  onChange={handleInputChange}
                  className={formErrors.message ? 'error' : ''}
                ></textarea>
                {formErrors.message && <p className="error-message">{formErrors.message}</p>}
              </div>
              <button type="submit">Send Message</button>
            </form>
          </div>
        </div>
      </section>

      <section className="faq-section">
        <div className="faq-container">
          <header className="faq-header">
            <h2>Frequently Asked Questions</h2>
            <p>Find answers to your most common queries. We've got you covered!</p>
          </header>
          <div className="faqs">
            {faqs.map((faq) => (
              <div className={`faq ${faqStates[faq.id] ? 'open' : ''}`} key={faq.id}>
                <div className="faq-question" onClick={() => toggleFaq(faq.id)}>
                  <span>{faq.question}</span>
                  <span className="faq-icon">{faqStates[faq.id] ? '-' : '+'}</span>
                </div>
                {faqStates[faq.id] && (
                  <div className="faq-answer">{faq.answer}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="footer-container">
          <div className="footer-section">
            <h2>RefactoAI</h2>
            <p>
              RefactoAI leverages AI to optimize your coding experience, making your development process faster, smarter, and more efficient.
            </p>
          </div>
          <div className="footer-section">
            <h3>Quick Links</h3>
            <ul>
              <li>Home</li>
              <li>Chatbot</li>
              <li>Documentation</li>
              <li>Login</li>
              <li>Sign Up</li>
            </ul>
          </div>
          <div className="footer-section">
            <h3>Resources</h3>
            <ul>
              <li>User Guide</li>
              <li>API Documentation</li>
              <li>FAQs</li>
              <li>Privacy Policy</li>
              <li>Terms of Service</li>
            </ul>
          </div>
          <div className="footer-section">
            <h3>Connect with Us</h3>
            <div className="social-icons">
              <a href="#">Facebook</a>
              <a href="#">Twitter</a>
              <a href="#">LinkedIn</a>
              <a href="#">GitHub</a>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>
            &copy; 2024 RefactoAI. All rights reserved. | Designed and developed as part of a Final Year Project at FAST-NUCES, Islamabad, under the supervision of Sir Sami Ullah.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
