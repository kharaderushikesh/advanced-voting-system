import streamlit as st
import hashlib
import json
import time
from datetime import datetime
from io import BytesIO
import face_recognition
import cv2
import numpy as np

# -------------------------
# Simple Blockchain Classes
# -------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash=""):
        self.index = index
        self.timestamp = timestamp
        self.data = data  # dict: {voter_aadhaar, candidate}
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, str(datetime.now()), {"message": "Genesis Block"}, "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            if current.hash != current.calculate_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True

    def has_voted(self, aadhaar):
        for block in self.chain:
            data = block.data
            if isinstance(data, dict) and data.get("voter_aadhaar") == aadhaar:
                return True
        return False

# -------------------------
# Session State Init
# -------------------------
if "voter_db" not in st.session_state:
    # Aadhaar -> voter details (now includes face_encoding as list for JSON)
    st.session_state.voter_db = {
        "123456789012": {"name": "Demo Voter 1", "age": 22, "address": "Test City", "face_enrolled": False, "face_encoding": None},
        "987654321098": {"name": "Demo Voter 2", "age": 30, "address": "Test Town", "face_enrolled": False, "face_encoding": None},
    }

if "blockchain" not in st.session_state:
    st.session_state.blockchain = Blockchain()

# -------------------------
# Helper Functions
# -------------------------
def extract_face_encoding(image_bytes):
    """Extract face encoding from image bytes using face_recognition."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_img)
        if face_locations:
            encodings = face_recognition.face_encodings(rgb_img, face_locations)
            return encodings[0].tolist()  # Return first encoding as list
        return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def add_new_voter(aadhaar, name, age, address, face_image_bytes=None):
    voter_db = st.session_state.voter_db

    if aadhaar in voter_db:
        return False, "Aadhaar already registered."

    if age < 18:
        return False, "Voter must be 18+."

    face_encoding = None
    if face_image_bytes:
        face_encoding = extract_face_encoding(face_image_bytes)
        if not face_encoding:
            return False, "No face detected in image. Please provide a clear face photo."

    voter_db[aadhaar] = {
        "name": name,
        "age": age,
        "address": address,
        "face_enrolled": face_encoding is not None,
        "face_encoding": face_encoding,
    }
    return True, "Voter added successfully with face enrollment."

def enroll_face(aadhaar, face_image_bytes):
    voter_db = st.session_state.voter_db

    if aadhaar not in voter_db:
        return False, "Aadhaar not found in voter database."

    face_encoding = extract_face_encoding(face_image_bytes)
    if not face_encoding:
        return False, "No face detected in image. Please provide a clear face photo."

    voter_db[aadhaar]["face_enrolled"] = True
    voter_db[aadhaar]["face_encoding"] = face_encoding
    return True, "Face enrolled successfully."

def verify_face_live(aadhaar, captured_image_bytes):
    """Verify live face against stored encoding."""
    voter_db = st.session_state.voter_db
    if aadhaar not in voter_db or not voter_db[aadhaar].get("face_encoding"):
        return False, "Face not enrolled for this voter."

    stored_encoding = np.array(voter_db[aadhaar]["face_encoding"])
    live_encoding = extract_face_encoding(captured_image_bytes)
    if not live_encoding:
        return False, "No face detected in live capture. Try again."

    # Compare encodings (threshold for match)
    matches = face_recognition.compare_faces([stored_encoding], np.array(live_encoding), tolerance=0.6)
    return matches[0], "Face verified." if matches[0] else "Face does not match enrolled face."

def cast_vote(aadhaar, candidate):
    voter_db = st.session_state.voter_db
    blockchain = st.session_state.blockchain

    # Check registration
    if aadhaar not in voter_db:
        return False, "Voter not registered."

    voter = voter_db[aadhaar]

    # Age check
    if voter["age"] < 18:
        return False, "Voter is under 18."

    # Face check (now real verification)
    if not voter.get("face_enrolled", False):
        return False, "Face not enrolled. Please enroll face first."

    # Duplicate vote check via blockchain
    if blockchain.has_voted(aadhaar):
        return False, "Voter has already voted."

    # Create vote block
    data = {
        "voter_aadhaar": aadhaar,
        "voter_name": voter["name"],
        "candidate": candidate,
        "time": str(datetime.now()),
    }
    new_block = Block(
        index=len(blockchain.chain),
        timestamp=str(datetime.now()),
        data=data,
        previous_hash=blockchain.get_latest_block().hash,
    )
    blockchain.add_block(new_block)
    return True, "Vote recorded on blockchain."

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Advanced Voting Machine Demo", layout="wide", page_icon="🗳️")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .block-expander {
        background-color: #e8f5e8;
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🗳️ Advanced Voting Machine Simulation")
st.caption("Aadhaar + Live Face Verification + Blockchain – Educational Demo Only")
st.markdown("---")

# Sidebar for quick stats
with st.sidebar:
    st.header("📊 Quick Stats")
    bc = st.session_state.blockchain
    st.metric("Total Voters", len(st.session_state.voter_db))
    st.metric("Total Votes Cast", len(bc.chain) - 1)  # Exclude genesis
    st.metric("Blockchain Valid", "✅ Yes" if bc.is_chain_valid() else "❌ No")
    st.markdown("**Demo Voters:** 123456789012, 987654321098")

tab_add, tab_enroll, tab_vote, tab_chain = st.tabs(
    ["➕ Add Voter", "📷 Enroll Face", "✅ Cast Vote", "⛓️ Blockchain View"]
)

# 1) Add Voter Tab
with tab_add:
    st.subheader("➕ Add a New Voter")
    st.markdown("Register a new voter with Aadhaar details and face enrollment (upload or live capture).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        aadhaar = st.text_input("Aadhaar Number (12 digits)", placeholder="e.g., 123456789012", key="add_aadhaar")
        name = st.text_input("Full Name", placeholder="e.g., John Doe", key="add_name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1, key="add_age")
        address = st.text_area("Address", placeholder="e.g., 123 Main St, City", key="add_address")
    
    with col2:
        st.markdown("**Face Enrollment**")
        face_option = st.radio("Choose image source:", ["Upload Image", "Live Capture"], key="add_face_option")
        
        face_image_bytes = None
        if face_option == "Upload Image":
            face_image = st.file_uploader("Select image file", type=["jpg", "jpeg", "png"], key="add_face_upload")
            if face_image:
                st.image(face_image, caption="Uploaded Image Preview", width=200)
                face_image_bytes = face_image.getvalue()
        elif face_option == "Live Capture":
            face_image = st.camera_input("Take a live photo", key="add_face_camera")
            if face_image:
                st.image(face_image, caption="Captured Image Preview", width=200)
                face_image_bytes = face_image.getvalue()
    
    msg_placeholder = st.empty()
    if st.button("➕ Add Voter", use_container_width=True):
        if not aadhaar or not name or not address or not face_image_bytes:
            st.error("❌ Please fill all fields and provide a face image (upload or live capture).")
        elif len(aadhaar) != 12 or not aadhaar.isdigit():
            st.error("❌ Aadhaar must be exactly 12 digits.")
        else:
            with st.spinner("Processing face..."):
                success, msg = add_new_voter(aadhaar, name, int(age), address, face_image_bytes)
            if success:
                msg_placeholder.success(f"✅ {msg}")
                time.sleep(3)
                msg_placeholder.empty()
                st.rerun()
            else:
                st.error(f"❌ {msg}")
    
    st.markdown("### 📋 Current Voter Database")
    with st.expander("View All Registered Voters"):
        if st.session_state.voter_db:
            st.json(st.session_state.voter_db)
        else:
            st.info("No voters registered yet.")

# 2) Enroll Face Tab
with tab_enroll:
    st.subheader("📷 Enroll or Update Face")
    st.markdown("Enroll a face for an existing voter (upload or live capture).")
    
    aadhaar_enroll = st.text_input("Aadhaar Number", placeholder="e.g., 123456789012", key="enroll_aadhaar")
    
    st.markdown("**Face Enrollment**")
    face_option2 = st.radio("Choose image source:", ["Upload Image", "Live Capture"], key="enroll_face_option")
    
    face_image_bytes2 = None
    if face_option2 == "Upload Image":
        face_image2 = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"], key="enroll_face_upload")
        if face_image2:
            st.image(face_image2, caption="Uploaded Image Preview", width=200)
            face_image_bytes2 = face_image2.getvalue()
    elif face_option2 == "Live Capture":
        face_image2 = st.camera_input("Take a live photo", key="enroll_face_camera")
        if face_image2:
            st.image(face_image2, caption="Captured Image Preview", width=200)
            face_image_bytes2 = face_image2.getvalue()
    
    msg_placeholder2 = st.empty()
    if st.button("📷 Enroll Face", use_container_width=True):
        if not aadhaar_enroll or not face_image_bytes2:
            st.error("❌ Please provide Aadhaar and a face image (upload or live capture).")
        elif len(aadhaar_enroll) != 12 or not aadhaar_enroll.isdigit():
            st.error("❌ Aadhaar must be exactly 12 digits.")
        else:
            with st.spinner("Processing face..."):
                success, msg = enroll_face(aadhaar_enroll, face_image_bytes2)
            if success:
                msg_placeholder2.success(f"✅ {msg}")
                time.sleep(3)
                msg_placeholder2.empty()
                st.rerun()
            else:
                st.error(f"❌ {msg}")

# 3) Cast Vote Tab
with tab_vote:
    st.subheader("✅ Cast Your Vote")
    st.markdown("Vote securely using Aadhaar and live face verification.")
    
    aadhaar_vote = st.text_input("Aadhaar Number", placeholder="e.g., 123456789012", key="vote_aadhaar")
    candidate = st.selectbox("Select Candidate", ["Candidate A", "Candidate B", "Candidate C"], key="vote_candidate")
    
    # Live face verification step
    if aadhaar_vote and len(aadhaar_vote) == 12 and aadhaar_vote.isdigit():
        st.markdown("### 📹 Live Face Verification")
        st.caption("Capture a live photo for verification.")
        captured_image = st.camera_input("Take a photo", key="live_face")
        
        if captured_image:
            st.image(captured_image, caption="Captured Image", width=200)
            msg_placeholder3 = st.empty()
            if st.button("🔍 Verify Face and Submit Vote", use_container_width=True):
                with st.spinner("Verifying face..."):
                    verified, msg = verify_face_live(aadhaar_vote, captured_image.getvalue())
                if verified:
                    success, vote_msg = cast_vote(aadhaar_vote, candidate)
                    if success:
                        msg_placeholder3.success(f"✅ {vote_msg}")
                        st.balloons()
                        time.sleep(3)
                        msg_placeholder3.empty()
                        st.rerun()
                    else:
                        st.error(f"❌ {vote_msg}")
                else:
                    st.error(f"❌ {msg}")
        else:
            st.info("Please capture a live photo to proceed.")
    else:
        st.info("Enter a valid 12-digit Aadhaar to enable face verification.")

# 4) Blockchain View Tab
with tab_chain:
    st.subheader("⛓️ Blockchain Ledger")
    st.markdown("View the immutable vote ledger. Each vote is a block.")
    
    bc = st.session_state.blockchain
    st.write(f"**Total Blocks:** {len(bc.chain)} (including Genesis)")
    st.write(f"**Chain Integrity:** {'✅ Valid' if bc.is_chain_valid() else '❌ Invalid'}")
    
    if len(bc.chain) > 1:
        st.markdown("### 🗳️ Vote Blocks")
        for block in bc.chain[1:]:  # Skip Genesis
            with st.expander(f"Block #{block.index} | Hash: {block.hash[:16]}...", expanded=False):
                st.markdown('<div class="block-expander">', unsafe_allow_html=True)
                st.write(f"**Timestamp:** {block.timestamp}")
                st.write(f"**Previous Hash:** {block.previous_hash}")
                st.write(f"**Nonce:** {block.nonce}")
                st.write("**Vote Data:**")
                st.json(block.data)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No votes cast yet. The ledger starts with the Genesis Block.")
    
    with st.expander("Genesis Block (Starting Point)"):
        genesis = bc.chain[0]
        st.write(f"**Index:** {genesis.index}")
        st.write(f"**Timestamp:** {genesis.timestamp}")
        st.write(f"**Data:** {genesis.data}")
        st.write(f"**Hash:** {genesis.hash}")

st.markdown("---")
st.caption("Built with Streamlit. This is a simulation – not for real-world use. Refresh the page to reset session state.")
