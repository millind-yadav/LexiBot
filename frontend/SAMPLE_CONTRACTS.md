# 📋 Sample Contract Text for Testing

## Contract A (Software Development Agreement):

```
SOFTWARE DEVELOPMENT AGREEMENT

This Software Development Agreement ("Agreement") is entered into on September 15, 2024, between TechCorp Solutions Inc., a Delaware corporation ("Company A"), and Digital Innovations LLC, a California limited liability company ("Company B").

SCOPE OF WORK:
Company A shall develop a custom customer relationship management (CRM) software application according to the specifications provided in Exhibit A. The deliverables include source code, documentation, and deployment support.

TIMELINE:
The project shall be completed within six (6) months from the effective date of this Agreement. Key milestones are defined in the project schedule attached as Exhibit B.

PAYMENT TERMS:
Total project cost is $150,000, payable in three installments:
- $50,000 upon signing (33%)
- $75,000 upon completion of development phase (50%)
- $25,000 upon final delivery and acceptance (17%)

INTELLECTUAL PROPERTY:
All intellectual property rights in the developed software shall remain with Company A. Company B receives a non-exclusive, perpetual license to use the software for its internal business operations only.

LIABILITY AND INDEMNIFICATION:
Company A's liability shall be limited to the total contract amount ($150,000). Company B agrees to indemnify Company A against any third-party claims arising from the use of the software.

TERMINATION:
Either party may terminate this agreement with thirty (30) days written notice. In case of termination by Company B, Company B shall pay 75% of the remaining contract value as liquidated damages.

WARRANTY:
Company A warrants the software will be free from material defects for ninety (90) days from delivery. Company A's sole obligation is to repair or replace defective software.

CONFIDENTIALITY:
Both parties agree to maintain confidentiality of proprietary information disclosed during the project for a period of five (5) years.

GOVERNING LAW:
This Agreement shall be governed by the laws of the State of Delaware.
```

## Contract B (Service Agreement):

```
PROFESSIONAL SERVICES AGREEMENT

This Professional Services Agreement ("Agreement") is effective September 20, 2024, between CloudTech Enterprises Corp., a New York corporation ("Service Provider"), and StartupCo Inc., a Texas corporation ("Client").

SERVICES:
Service Provider will provide web application development services including design, development, testing, and maintenance of a customer portal system as detailed in Statement of Work #001.

PROJECT DURATION:
Services will be provided over a four (4) month period commencing October 1, 2024. Extensions require written approval from both parties.

COMPENSATION:
Client agrees to pay $120,000 for services rendered:
- $36,000 upon contract execution (30%)
- $48,000 at 50% project completion (40%)
- $36,000 upon final delivery (30%)

OWNERSHIP RIGHTS:
Client will own all custom-developed code and related intellectual property upon full payment. Service Provider retains rights to any pre-existing tools, frameworks, or methodologies.

LIMITATION OF LIABILITY:
Service Provider's liability is capped at the amount paid under this Agreement. Neither party shall be liable for consequential, indirect, or punitive damages.

CONTRACT TERMINATION:
Client may terminate for convenience with sixty (60) days notice and payment of 25% penalty on remaining work. Service Provider may terminate for non-payment after 15 days notice.

QUALITY ASSURANCE:
Service Provider provides a twelve (12) month warranty on delivered software, including bug fixes and minor enhancements at no additional cost.

DATA PROTECTION:
Service Provider agrees to implement appropriate security measures and comply with applicable data protection regulations including GDPR and CCPA.

DISPUTE RESOLUTION:
Any disputes will be resolved through binding arbitration under the rules of the American Arbitration Association.

FORCE MAJEURE:
Neither party will be liable for delays caused by circumstances beyond their reasonable control, including natural disasters, government actions, or pandemics.
```

## 🧪 Test Instructions:

1. **Start your server:**
   ```bash
   cd /Users/milind/Documents/lexibot-agent-project/frontend
   npm run dev
   ```

2. **Open browser:**
   ```
   http://localhost:3000/legal
   ```

3. **Test Contract Analysis:**
   - Copy "Contract A" text above
   - Paste into the analyzer
   - Select "Comprehensive Analysis"
   - Click "Analyze Contract"
   - Should see detailed analysis with risk assessment

4. **Test Contract Comparison:**
   - Copy "Contract A" into left side
   - Copy "Contract B" into right side  
   - Click "Compare Contracts"
   - Should see detailed comparison highlighting differences

## ✨ Expected Results:

- **Risk badges** (High/Medium/Low)
- **Confidence scores** (~90%+)
- **Processing times** (~2-4 seconds)
- **Detailed analysis** for each contract
- **Comparison highlights** key differences
- **Professional UI** with proper styling