"""
API routes for the research field similarity application.
"""

import logging
from flask import Blueprint, request, jsonify, send_file
from typing import Dict, List, Any

from services.data_service import DataService
from services.similarity.final_calculation import FieldSimilarityService
from config import SIMILARITY_FILE

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
data_service = DataService()
field_similarity_service = FieldSimilarityService()

# Load initial data to set up field mappings
nested_data, _ = data_service.load_data()
field_similarity_service.update_field_mappings(nested_data)


@api_bp.route('/add_field', methods=['POST'])
def add_field():
    """Add a new field and calculate similarities."""
    # Load current data
    nested_data, similarities = data_service.load_data()
    
    # Get form data
    field_data = {
        "name": request.form.get('name'),
        "group": request.form.get('group'),
        "subgroup": request.form.get('subgroup'),
        "definition": request.form.get('definition'),
        "methodologies": request.form.get('methodologies'),
        "applications": request.form.get('applications'),
        "technologies": request.form.get('technologies'),
        "challenges": request.form.get('challenges'),
        "future_directions": request.form.get('future_directions')
    }
    
    # Validate required fields
    if not field_data["name"] or not field_data["group"] or not field_data["subgroup"]:
        return jsonify({"error": "Name, group, and subgroup are required"}), 400
    
    # Check if field name already exists
    field_names = data_service.get_all_field_names(nested_data)
    if field_data["name"] in field_names:
        return jsonify({"error": "Field name already exists"}), 400
    
    # Add field to data
    nested_data, new_field = data_service.add_field_to_data(nested_data, field_data)
    
    # Update field mappings
    field_similarity_service.update_field_mappings(nested_data)
    
    # Calculate new similarities
    updated_similarities = field_similarity_service.calculate_new_similarities(
        nested_data, similarities, new_field)
    
    # Save updated data
    if data_service.save_data(nested_data, updated_similarities):
        return jsonify({
            "success": True,
            "message": "Field added and similarities calculated",
            "download_ready": True
        })
    else:
        return jsonify({"error": "Error saving data"}), 500


@api_bp.route('/get_subgroups', methods=['GET'])
def get_subgroups():
    """Get subgroups for a specific group."""
    group = request.args.get('group')
    
    if not group:
        return jsonify({"error": "Group parameter is required"}), 400
    
    nested_data, _ = data_service.load_data()
    _, subgroups = data_service.get_all_groups_and_subgroups(nested_data)
    
    return jsonify({
        "success": True,
        "subgroups": subgroups.get(group, [])
    })


@api_bp.route('/get_similarity', methods=['GET'])
def get_similarity():
    """Get similarity between two fields."""
    field1 = request.args.get('field1')
    field2 = request.args.get('field2')
    
    if not field1 or not field2:
        return jsonify({"error": "Both field1 and field2 parameters are required"}), 400
    
    nested_data, similarities = data_service.load_data()
    
    # Get field data
    field1_data = data_service.get_field_data(nested_data, field1)
    field2_data = data_service.get_field_data(nested_data, field2)
    
    if not field1_data or not field2_data:
        return jsonify({"error": "One or both fields not found"}), 404
    
    # Find similarity
    similarity = data_service.find_similarity(similarities, field1, field2)
    
    if similarity is None:
        # If similarity is not found in the database, calculate it on the fly
        try:
            similarity = field_similarity_service.compare_fields(field1_data, field2_data)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return jsonify({"error": "Error calculating similarity"}), 500
    
    return jsonify({
        "success": True,
        "field1": field1,
        "field2": field2,
        "similarity": similarity,
        "field1_data": field1_data,
        "field2_data": field2_data
    })


@api_bp.route('/get_all_similarities_for_field', methods=['GET'])
def get_all_similarities_for_field():
    """Get similarities between one field and all other fields."""
    field_name = request.args.get('field')
    
    if not field_name:
        return jsonify({"error": "Field parameter is required"}), 400
    
    nested_data, similarities = data_service.load_data()
    
    # Get the source field data
    source_field_data = data_service.get_field_data(nested_data, field_name)
    
    if not source_field_data:
        return jsonify({"error": f"Field '{field_name}' not found"}), 404
    
    # Extract all other field names
    all_field_names = data_service.get_all_field_names(nested_data)
    
    # Find all similarities involving this field
    field_similarities = []
    for other_field in all_field_names:
        if other_field == field_name:
            continue  # Skip self-comparison
            
        # Find similarity between these fields
        similarity_score = data_service.find_similarity(similarities, field_name, other_field)
        
        if similarity_score is not None:
            # Get the other field's data
            other_field_data = data_service.get_field_data(nested_data, other_field)
            
            # Get group/subgroup info
            group, subgroup = data_service.get_field_group_info(nested_data, other_field)
            
            # Add to results
            field_similarities.append({
                "field": other_field,
                "similarity": similarity_score,
                "group": group or "",
                "subgroup": subgroup or "",
                "field_data": other_field_data
            })
    
    # Sort by similarity (highest first)
    field_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return jsonify({
        "success": True,
        "field": field_name,
        "source_field_data": source_field_data,
        "similarities": field_similarities
    })


@api_bp.route('/download_similarities')
def download_similarities():
    """Download similarities file."""
    return send_file(SIMILARITY_FILE, as_attachment=True)


@api_bp.route('/test')
def test():
    """Test route to check if data is loading correctly."""
    nested_data, similarities = data_service.load_data()
    field_names = data_service.get_all_field_names(nested_data)
    groups, subgroups = data_service.get_all_groups_and_subgroups(nested_data)
    
    return jsonify({
        "success": True,
        "data_loaded": True,
        "field_count": len(field_names),
        "group_count": len(groups),
        "similarity_count": len(similarities),
        "fields": field_names if field_names else [],
        "groups": groups if groups else []
    })


# @api_bp.route('/recalculate_similarities', methods=['POST'])
# def recalculate_similarities():
#     """Recalculate similarities for all fields."""
#     nested_data, _ = data_service.load_data()
    
#     # Recalculate similarities
#     try:
#         updated_similarities = field_similarity_service.calculate_all_similarities(nested_data)
#     except Exception as e:
#         logger.error(f"Error recalculating similarities: {e}")
#         return jsonify({"error": "Error recalculating similarities"}), 500
    
#     # Save updated data
#     if data_service.save_data(nested_data, updated_similarities):
#         return jsonify({
#             "success": True,
#             "message": "Similarities recalculated and saved",
#             "download_ready": True
#         })
#     else:
#         return jsonify({"error": "Error saving data"}), 500